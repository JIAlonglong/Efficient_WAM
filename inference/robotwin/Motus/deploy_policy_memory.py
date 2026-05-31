# Motus Policy with Memory Bank for RoboTwin Evaluation
# Extends deploy_policy.py with ActionMemoryInjector + FIFO memory bank.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add model paths (insert at beginning to avoid conflicts with project root's utils)
_policy_dir = str(Path(__file__).parent)
if _policy_dir not in sys.path:
    sys.path.insert(0, _policy_dir)
_models_dir = str(Path(__file__).parent / "models")
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ActionMemoryInjector (same as train_injector.py)
# ---------------------------------------------------------------------------

class ActionMemoryInjector(nn.Module):
    """Inject memory into action tokens via cross-attention."""
    def __init__(self, action_dim: int = 1024, memory_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = action_dim // num_heads

        self.q_proj = nn.Linear(action_dim, action_dim)
        self.k_proj = nn.Linear(memory_dim, action_dim)
        self.v_proj = nn.Linear(memory_dim, action_dim)
        self.out_proj = nn.Linear(action_dim, action_dim)
        self.norm_q = nn.LayerNorm(action_dim)
        self.norm_memory = nn.LayerNorm(memory_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, action_tokens: torch.Tensor, memory_features: torch.Tensor) -> torch.Tensor:
        B, S_a, D = action_tokens.shape
        K, S_m, D_m = memory_features.shape[1], memory_features.shape[2], memory_features.shape[3]
        input_dtype = action_tokens.dtype

        memory_flat = memory_features.reshape(B, K * S_m, D_m)

        q = self.norm_q(action_tokens.float())
        memory_normed = self.norm_memory(memory_flat.float())

        Q = self.q_proj(q).view(B, S_a, self.num_heads, self.head_dim).transpose(1, 2)
        K_proj = self.k_proj(memory_normed).view(B, K * S_m, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(memory_normed).view(B, K * S_m, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(Q, K_proj, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S_a, D)
        output = self.out_proj(attn_output)

        return (action_tokens + torch.sigmoid(self.gate) * output).to(input_dtype)


# ---------------------------------------------------------------------------
# RecencyWeightedRetriever (for soft retrieval variant)
# ---------------------------------------------------------------------------

class RecencyWeightedRetriever(nn.Module):
    """Recency-weighted soft retrieval."""
    def __init__(self, motion_dim: int = 256, init_decay: float = 1.0):
        super().__init__()
        self.motion_dim = motion_dim
        self.log_decay = nn.Parameter(torch.tensor(float(init_decay)).log())

    @property
    def decay(self) -> torch.Tensor:
        return torch.exp(self.log_decay)

    def forward(self, query_motion, memory_bank):
        stored_vlm = memory_bank.get_all_vlm()
        stored_motion = memory_bank.get_all_motion()
        if stored_vlm is None:
            return None, None

        B, K = stored_motion.shape[:2]
        motion_scores = F.cosine_similarity(query_motion.unsqueeze(1), stored_motion, dim=-1)

        timestamps = torch.tensor(
            memory_bank.timestamps, dtype=torch.float32, device=query_motion.device
        ).unsqueeze(0).expand(B, -1)
        current_time = timestamps.max(dim=-1, keepdim=True).values
        age = current_time - timestamps
        recency_bias = -self.decay * age
        combined = motion_scores + recency_bias
        weights = F.softmax(combined, dim=-1)

        weighted_vlm = stored_vlm * weights.unsqueeze(-1).unsqueeze(-1)
        return weighted_vlm, weights


# ---------------------------------------------------------------------------
# FIFO Memory Bank
# ---------------------------------------------------------------------------

class FIFOMemoryBank:
    def __init__(self, max_size: int = 5, device: str = "cuda"):
        self.max_size = max_size
        self.device = device
        self.clear()

    def clear(self):
        self.vlm_features: List[torch.Tensor] = []
        self.motion_features: List[torch.Tensor] = []
        self.timestamps: List[int] = []

    @property
    def size(self) -> int:
        return len(self.vlm_features)

    def add(self, vlm_features: torch.Tensor, motion_features: torch.Tensor, timestamp: int):
        if self.size >= self.max_size:
            self.vlm_features.pop(0)
            self.motion_features.pop(0)
            self.timestamps.pop(0)
        self.vlm_features.append(vlm_features.detach().clone())
        self.motion_features.append(motion_features.detach().clone())
        self.timestamps.append(timestamp)

    def get_all_vlm(self) -> Optional[torch.Tensor]:
        if self.size == 0:
            return None
        B, seq_len, dim = self.vlm_features[0].shape
        features = torch.zeros(B, self.max_size, seq_len, dim, device=self.device, dtype=self.vlm_features[0].dtype)
        for i, feat in enumerate(self.vlm_features):
            features[:, i] = feat
        return features

    def get_all_motion(self) -> Optional[torch.Tensor]:
        if self.size == 0:
            return None
        B, dim = self.motion_features[0].shape
        features = torch.zeros(B, self.max_size, dim, device=self.device, dtype=self.motion_features[0].dtype)
        for i, feat in enumerate(self.motion_features):
            features[:, i] = feat
        return features


# ---------------------------------------------------------------------------
# Motion Feature Extractor
# ---------------------------------------------------------------------------

class SimpleMotionExtractor:
    def __init__(self, feature_dim: int = 256, device: str = "cuda"):
        self.feature_dim = feature_dim
        self.device = device
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, feature_dim),
        ).to(device)
        self.encoder.eval()

    @torch.no_grad()
    def extract(self, current_frame: torch.Tensor, previous_frame: torch.Tensor) -> torch.Tensor:
        diff = current_frame - previous_frame
        diff_mag = torch.sqrt(diff.pow(2).mean(dim=1, keepdim=True) + 1e-8)
        return self.encoder(diff_mag)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@torch.no_grad()
def retrieve_memory(
    query_motion: torch.Tensor,
    memory_bank: FIFOMemoryBank,
    top_k: int = 5,
) -> Optional[torch.Tensor]:
    """Top-k retrieval by motion similarity."""
    if memory_bank.size == 0:
        return None

    stored_vlm = memory_bank.get_all_vlm()
    stored_motion = memory_bank.get_all_motion()

    scores = F.cosine_similarity(query_motion.unsqueeze(1), stored_motion, dim=-1)
    actual_k = min(top_k, memory_bank.size)
    _, top_k_indices = scores.topk(actual_k, dim=1)

    retrieved_vlm = torch.gather(
        stored_vlm, 1,
        top_k_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, stored_vlm.shape[2], stored_vlm.shape[3]),
    )
    return retrieved_vlm


# ---------------------------------------------------------------------------
# Memory-augmented inference step (monkey-patch for Motus model)
# ---------------------------------------------------------------------------

def make_memory_inference_step(injector, memory_bank, motion_extractor, top_k=5, use_soft=False, retriever=None):
    """Create a memory-augmented inference_step function.

    Returns a function that can replace model.inference_step.
    """
    # Get the original inference_step
    from functools import partial

    def memory_inference_step(
        self,
        first_frame: torch.Tensor,
        state: torch.Tensor = None,
        num_inference_steps: int = 50,
        language_embeddings: Optional[List[torch.Tensor]] = None,
        vlm_inputs: Optional[List] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = first_frame.shape[0]

        language_embeddings = [emb.to(self.device).to(self.dtype) for emb in language_embeddings]
        state = state.to(self.device).to(self.dtype)
        first_frame = first_frame.to(self.device).to(self.dtype)

        # 1. Encode condition frame
        first_frame_norm = (first_frame * 2.0 - 1.0).unsqueeze(2)
        with torch.no_grad():
            condition_frame_latent = self.video_model.encode_video(first_frame_norm.to(self.dtype))

        B, C_latent, f_latent, H_latent, W_latent = condition_frame_latent.shape
        num_total_latent_frames = 1 + self.config.num_video_frames // 4
        video_latent = torch.randn((B, C_latent, num_total_latent_frames, H_latent, W_latent),
                                   device=self.device, dtype=self.dtype)
        video_latent[:, :, 0:1] = condition_frame_latent
        action_shape = (B, self.config.action_chunk_size, self.config.action_dim)
        action_latent = torch.randn(action_shape, device=self.device, dtype=self.dtype)

        # 2. Understanding Expert features
        und_tokens = self.und_module.extract_und_features(vlm_inputs)
        processed_t5_context = self.video_module.preprocess_t5_embeddings(language_embeddings)

        # 3. Prepare memory features
        mem_features = None
        if memory_bank.size > 0:
            query_motion = torch.zeros(B, motion_extractor.feature_dim, device=self.device)
            if use_soft and retriever is not None:
                retrieved, _ = retriever(query_motion, memory_bank)
            else:
                retrieved = retrieve_memory(query_motion, memory_bank, top_k=top_k)
            if retrieved is not None:
                mem_features = retrieved.to(self.device).float()

        # 4. Denoising loop
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device, dtype=self.dtype)
        for i in range(num_inference_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t
            video_t_scaled = (t * 1000).expand(B).to(self.dtype)
            action_t_scaled = (t * 1000).expand(B).to(self.dtype)

            video_tokens = self.video_module.prepare_input(video_latent.to(self.dtype))
            state_tokens = state.unsqueeze(1).to(self.dtype)
            registers = self.action_expert.registers.expand(B, -1, -1)
            action_tokens = self.action_expert.input_encoder(state_tokens, action_latent, registers)

            und_tokens = self.und_module.extract_und_features(vlm_inputs)

            with torch.autocast(device_type="cuda", dtype=self.video_model.precision):
                video_head_time_emb, video_adaln_params = self.video_module.get_time_embedding(
                    video_t_scaled, video_tokens.shape[1])
                action_head_time_emb, action_adaln_params = self.action_module.get_time_embedding(
                    action_t_scaled, action_tokens.shape[1])

                for layer_idx in range(self.config.num_layers):
                    video_adaln_modulation = self.video_module.compute_adaln_modulation(
                        video_adaln_params, layer_idx)
                    action_adaln_modulation = self.action_module.compute_adaln_modulation(
                        action_adaln_params, layer_idx)

                    video_tokens, action_tokens, und_tokens = self.video_module.process_joint_attention(
                        video_tokens, action_tokens,
                        video_adaln_modulation, action_adaln_modulation,
                        layer_idx, self.action_expert.blocks[layer_idx],
                        und_tokens, self.und_expert.blocks[layer_idx],
                    )

                    video_tokens = self.video_module.process_cross_attention(
                        video_tokens, video_adaln_params, layer_idx, processed_t5_context)
                    video_tokens = self.video_module.process_ffn(
                        video_tokens, video_adaln_modulation, layer_idx)
                    action_tokens = self.action_module.process_ffn(
                        action_tokens, action_adaln_modulation, layer_idx)
                    und_tokens = self.und_module.process_ffn(und_tokens, layer_idx)

                # === Memory injection into action tokens ===
                if mem_features is not None:
                    action_tokens = injector(action_tokens.float(), mem_features).to(self.dtype)

                video_velocity = self.video_module.apply_output_head(
                    video_tokens, video_head_time_emb)
                action_pred_full = self.action_expert.decoder(action_tokens, action_head_time_emb)
                action_velocity = action_pred_full[:, 1:-self.action_expert.config.num_registers, :]

                video_latent = video_latent + video_velocity * dt
                action_latent = action_latent + action_velocity * dt
                video_latent[:, :, 0:1] = condition_frame_latent

        # Decode video
        video_latent_5d = video_latent.unsqueeze(0) if video_latent.dim() == 4 else video_latent
        with torch.no_grad():
            predicted_frames = self.video_model.decode_video(video_latent_5d.to(self.dtype))

        # Denormalize actions
        action_flat = action_latent.reshape(-1, action_latent.shape[-1])
        action_range = self.action_max - self.action_min
        predicted_actions = action_flat * action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        predicted_actions = predicted_actions.reshape(action_latent.shape)

        return predicted_frames, predicted_actions

    return memory_inference_step


# ---------------------------------------------------------------------------
# MotusPolicy with Memory
# ---------------------------------------------------------------------------

class MotusPolicyMemory:
    """Motus Policy with memory-augmented inference."""

    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str,
                 injector_checkpoint: str, bank_size: int = 5, top_k: int = 5,
                 use_soft: bool = False, device: str = "cuda",
                 log_dir: Optional[str] = None, task_name: Optional[str] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        self.bank_size = bank_size
        self.top_k = top_k
        self.use_soft = use_soft

        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)

        # Initialize model
        self.model = self._load_model()

        # Load injector (and optionally retriever)
        self.injector, self.retriever = self._load_injector(injector_checkpoint)

        # Initialize memory components
        self.memory_bank = FIFOMemoryBank(max_size=bank_size, device=device)
        self.motion_extractor = SimpleMotionExtractor(device=device)

        # Monkey-patch inference_step with memory-augmented version
        import types
        mem_step = make_memory_inference_step(
            self.injector, self.memory_bank, self.motion_extractor,
            top_k=top_k, use_soft=use_soft, retriever=self.retriever,
        )
        self.model.inference_step = types.MethodType(mem_step, self.model)

        # Initialize T5 encoder
        self.t5_encoder = T5EncoderModel(
            text_len=512, dtype=torch.bfloat16, device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)

        # Caches
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None
        self.prev_frame = None
        self.step_count = 0
        self.episode_count = 0

        # Load normalization stats
        self._load_normalization_stats()

        # Image saving
        self.save_images = True
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("MotusPolicyMemory initialized successfully")

    def _load_model(self) -> Motus:
        config = self._create_model_config()
        model = Motus(config)
        model = model.to(self.device)
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        model.eval()
        return model

    def _create_model_config(self) -> MotusConfig:
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']
        vae_path = os.path.join(self.wan_path, "Wan2.2_VAE.pth")
        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        return MotusConfig(
            wan_checkpoint_path=self.wan_path, vae_path=vae_path,
            wan_config_path=self.wan_path, video_precision='bfloat16',
            vlm_checkpoint_path=self.vlm_path,
            und_expert_hidden_size=512, und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5, und_layers_to_extract=None,
            vlm_adapter_input_dim=2048, vlm_adapter_projector_type="mlp3x_silu",
            num_layers=30, action_state_dim=common['state_dim'],
            action_dim=common['action_dim'], action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier, action_expert_norm_eps=1e-6,
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0, action_loss_weight=1.0,
            batch_size=1, video_height=common['video_height'],
            video_width=common['video_width'],
            load_pretrained_backbones=False, training_mode='finetune',
        )

    def _load_injector(self, injector_checkpoint: str):
        """Load trained injector (and optionally retriever) from checkpoint."""
        ckpt = torch.load(injector_checkpoint, map_location=self.device)

        injector = ActionMemoryInjector(action_dim=1024, memory_dim=512)
        injector.load_state_dict(ckpt['injector_state_dict'])
        injector = injector.to(self.device)
        injector.eval()

        retriever = None
        if self.use_soft and 'retriever_state_dict' in ckpt:
            retriever = RecencyWeightedRetriever(motion_dim=256)
            retriever.load_state_dict(ckpt['retriever_state_dict'])
            retriever = retriever.to(self.device)
            retriever.eval()
            logger.info(f"Loaded retriever with decay={retriever.decay.item():.4f}")

        logger.info(f"Loaded injector from {injector_checkpoint}")
        return injector, retriever

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction

    def update_obs(self, observation: Dict[str, Any]):
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                left_img_resized = cv2.resize(left_img, (160, 120))
                right_img_resized = cv2.resize(right_img, (160, 120))
                bottom_row = np.concatenate([left_img_resized, right_img_resized], axis=1)
                image = np.concatenate([head_img, bottom_row], axis=0)
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)

        self.obs_cache.append(image_tensor.to(self.device))

        state = observation['joint_action']['vector']
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self._normalize_actions(self.current_state).to(self.device)

    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache")
        if self.current_state is None:
            raise ValueError("No robot state available")

        current_frame = self.obs_cache[-1]

        # Update memory bank with current frame
        with torch.no_grad():
            vlm_inputs = self._preprocess_vlm_messages(
                self.current_instruction,
                self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
            )
            # Extract VLM features for memory
            hist_und = self.model.und_module.extract_und_features([vlm_inputs])

            # Extract motion features
            if self.prev_frame is not None:
                motion_feat = self.motion_extractor.extract(current_frame, self.prev_frame)
            else:
                motion_feat = torch.zeros(1, self.motion_extractor.feature_dim, device=self.device)

            self.memory_bank.add(hist_und, motion_feat, timestamp=self.step_count)
            self.prev_frame = current_frame.clone()

        # Encode instruction
        scene_prefix = ("The whole scene is in a realistic, industrial art style with three views: "
                        "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
                        "The aloha robot is currently performing the following task: ")
        instruction = f"{scene_prefix}{self.current_instruction}"
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Run inference (memory-augmented via monkey-patch)
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                condition_frame_viz = current_frame.squeeze(0)
                predicted_frames_viz = predicted_frames_viz.squeeze(0)
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def _tensor_to_pil_image(self, tensor_chw):
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction, image):
        messages = [{'role': 'user', 'content': [
            {'type': 'text', 'text': instruction}, {'type': 'image', 'image': image}
        ]}]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device),
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        stat_path = Path(__file__).parent / 'utils' / 'stat.json'
        import json as _json
        with open(stat_path, 'r') as f:
            stat_data = _json.load(f)
        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y):
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)

    def _save_frame_grid(self, condition_frame, predicted_frames):
        if not self.save_images:
            return
        try:
            def tensor_to_numpy(tensor):
                if tensor.dim() == 3:
                    tensor = tensor.permute(1, 2, 0)
                tensor = tensor.detach().cpu().float()
                tensor = torch.clamp(tensor, 0, 1)
                return (tensor.numpy() * 255).astype(np.uint8)

            condition_np = tensor_to_numpy(condition_frame)
            predicted_np = [tensor_to_numpy(predicted_frames[i]) for i in range(predicted_frames.shape[0])]
            while len(predicted_np) < 4:
                predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
            all_frames = [condition_np] + predicted_np[:4]
            grid_image = np.concatenate(all_frames, axis=1)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            Image.fromarray(grid_image).save(save_path)
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")


# ---------------------------------------------------------------------------
# RoboTwin interface functions
# ---------------------------------------------------------------------------

def encode_obs(observation):
    return observation


def get_model(usr_args):
    """Initialize MotusPolicyMemory."""
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')
    vlm_path = usr_args.get('vlm_path')
    injector_checkpoint = usr_args.get('injector_checkpoint')
    bank_size = usr_args.get('bank_size', 5)
    top_k = usr_args.get('top_k', 5)
    use_soft = usr_args.get('use_soft', False)

    if not wan_path:
        raise ValueError("wan_path not provided")
    if not vlm_path:
        raise ValueError("vlm_path not provided")
    if not injector_checkpoint:
        raise ValueError("injector_checkpoint not provided")

    policy_dir = Path(__file__).parent
    config_path = policy_dir / "utils" / "robotwin.yml"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = MotusPolicyMemory(
        checkpoint_path=checkpoint_path, wan_path=wan_path, vlm_path=vlm_path,
        config_path=str(config_path), injector_checkpoint=injector_checkpoint,
        bank_size=bank_size, top_k=top_k, use_soft=use_soft,
        device=device, log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name'),
    )
    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function — same interface as deploy_policy.py."""
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)
    actions = model.get_action()
    for action in actions:
        TASK_ENV.take_action(action, action_type='qpos')


def reset_model(model):
    """Reset model cache and memory bank at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.memory_bank.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.prev_frame = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
