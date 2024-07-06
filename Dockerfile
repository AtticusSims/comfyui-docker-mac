FROM --platform=linux/arm64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

ARG USE_PERSISTENT_DATA

RUN apt-get update && apt-get install -y \
    git libgl1 libglib2.0-0 \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git git-lfs \
    ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /code


# User
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Pyenv
RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH
ARG PYTHON_VERSION=3.10.12

# Python
RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    datasets \
    huggingface-hub "protobuf<4" "click<8.1"

# PyTorch installation for Mac M1/M2
RUN pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# Set environment variable for MPS fallback
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify MPS availability
RUN python3 -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"


# Set the working directory to /data if USE_PERSISTENT_DATA is set, otherwise set to $HOME/app
WORKDIR $HOME/app


RUN git clone https://github.com/comfyanonymous/ComfyUI . && \pip install xformers!=0.0.24 --no-cache-dir -r requirements.txt 

# instal custom nodes
RUN echo "Installing custom nodes..." 
RUN pip install -U onnxruntime


RUN cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git
RUN cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack && cd ComfyUI-Impact-Pack && python install.py
# RUN cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack && cd ComfyUI-Inspire-Pack && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation && cd ComfyUI-Frame-Interpolation && python install.py
# RUN cd custom_nodes && git clone https://github.com/Fannovel16/ComfyUI-Video-Matting && cd ComfyUI-Video-Matting && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/BlenderNeko/ComfyUI_Cutoff
# RUN cd custom_nodes && git clone https://github.com/WASasquatch/PPF_Noise_ComfyUI && cd PPF_Noise_ComfyUI && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/WASasquatch/PowerNoiseSuite && cd PowerNoiseSuite && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/Jordach/comfy-plasma
# RUN cd custom_nodes && git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes
# RUN cd custom_nodes && git clone https://github.com/space-nuko/ComfyUI-OpenPose-Editor
# RUN cd custom_nodes && git clone https://github.com/twri/sdxl_prompt_styler
# RUN cd custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved
# RUN cd custom_nodes && git clone https://github.com/AIrjen/OneButtonPrompt
# RUN cd custom_nodes && git clone https://github.com/WASasquatch/was-node-suite-comfyui && cd was-node-suite-comfyui && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/cubiq/ComfyUI_essentials
# RUN cd custom_nodes && git clone https://github.com/crystian/ComfyUI-Crystools && cd ComfyUI-Crystools && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive
# RUN cd custom_nodes && git clone https://github.com/gokayfem/ComfyUI_VLM_nodes && cd ComfyUI_VLM_nodes && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/Fannovel16/comfyui_controlnet_aux && cd comfyui_controlnet_aux && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/Stability-AI/stability-ComfyUI-nodes && cd stability-ComfyUI-nodes && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/jags111/efficiency-nodes-comfyui && cd efficiency-nodes-comfyui && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts
# RUN cd custom_nodes && git clone https://github.com/WASasquatch/FreeU_Advanced
# RUN cd custom_nodes && git clone https://github.com/city96/SD-Advanced-Noise
# RUN cd custom_nodes && git clone https://github.com/kadirnar/ComfyUI_Custom_Nodes_AlekPet
# RUN cd custom_nodes && git clone https://github.com/sipherxyz/comfyui-art-venture && cd comfyui-art-venture && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/evanspearman/ComfyMath && cd ComfyMath && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/Gourieff/comfyui-reactor-node && cd comfyui-reactor-node && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/rgthree/rgthree-comfy && cd rgthree-comfy && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/giriss/comfy-image-saver && cd comfy-image-saver && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/gokayfem/ComfyUI-Depth-Visualization && cd ComfyUI-Depth-Visualization && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/kohya-ss/ControlNet-LLLite-ComfyUI
# RUN cd custom_nodes && git clone https://github.com/gokayfem/ComfyUI-Dream-Interpreter && cd ComfyUI-Dream-Interpreter && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus
# RUN cd custom_nodes && git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet && cd ComfyUI-Advanced-ControlNet && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/Acly/comfyui-inpaint-nodes 
# RUN cd custom_nodes && git clone https://github.com/chflame163/ComfyUI_LayerStyle  && cd ComfyUI_LayerStyle && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92
# RUN cd custom_nodes && git clone https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes
# RUN cd custom_nodes && git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes
# RUN cd custom_nodes && git clone https://github.com/jags111/ComfyUI_Jags_VectorMagic
# RUN cd custom_nodes && git clone https://github.com/melMass/comfy_mtb  && cd comfy_mtb && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/AuroBit/ComfyUI-OOTDiffusion  && cd ComfyUI-OOTDiffusion && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/kijai/ComfyUI-KJNodes && cd ComfyUI-KJNodes && pip install -r requirements.txt 
# RUN cd custom_nodes && git clone https://github.com/kijai/ComfyUI-SUPIR && cd ComfyUI-SUPIR && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/kijai/ComfyUI-depth-fm && cd ComfyUI-depth-fm && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/viperyl/ComfyUI-BiRefNet && cd ComfyUI-BiRefNet && pip install -r requirements.txt
# RUN cd custom_nodes && git clone https://github.com/gokayfem/ComfyUI-Texture-Simple
# RUN cd custom_nodes && git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-APISR && cd ComfyUI-APISR && pip install -r requirements.txt

# RUN echo "Downloading checkpoints..."  
# RUN wget -c https://huggingface.co/kadirnar/Black-Hole/resolve/main/tachyon.safetensors -P ./models/checkpoints/ 
# RUN wget -c https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning/resolve/main/RealVisXL_V4.0_Lightning.safetensors -P ./models/checkpoints/ 
# RUN wget -c https://huggingface.co/Lykon/dreamshaper-xl-lightning/resolve/main/DreamShaperXL_Lightning.safetensors -P ./models/checkpoints/ 
RUN wget -c https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors -P ./models/checkpoints/ 
# RUN wget -c https://huggingface.co/ckpt/stable-diffusion-3-medium/resolve/main/sd3_medium.safetensors -P ./models/checkpoints/ 

# RUN echo "Downloading AnimateDiff Models..."  

# RUN wget -c https://huggingface.co/hotshotco/Hotshot-XL/resolve/main/hsxl_temporal_layers.f16.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/ 
# RUN wget -c https://huggingface.co/hotshotco/SDXL-512/resolve/main/hsxl_base_1.0.f16.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_rgb.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_sparsectrl_scribble.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/ 
# RUN wget -c https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_8step_comfyui.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/  
# RUN wget -c https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_4step_comfyui.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
# RUN wget -c https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_2step_comfyui.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/
# RUN wget -c https://huggingface.co/ByteDance/AnimateDiff-Lightning/resolve/main/animatediff_lightning_1step_comfyui.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/models/

# RUN echo "Downloading Vae..."  

# RUN wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -P ./models/vae/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/taesdxl.safetensors -P ./models/vae/ 
# RUN wget -c https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors -P ./models/vae/ 

# RUN echo "Downloading Controlnet..."  

# RUN wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors -P ./models/controlnet/ 
# RUN wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors -P ./models/controlnet/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet-sdxl-canny-mid.safetensors -P ./models/controlnet/
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet-sdxl-depth-mid.safetensors -P ./models/controlnet/
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet_scribble_sd15.safetensors -P ./models/controlnet/
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/control_v11p_sd15s2_lineart_anime.safetensors -P ./models/controlnet/
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/control_v11p_sd15_lineart.safetensors -P ./models/controlnet/
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/control_v11p_sd15_canny_fp16.safetensors -P ./models/controlnet/
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/controlnet_depth_sd15.safetensors -P ./models/controlnet/
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_scribble_anime.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_blur_anime_beta.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_canny.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_canny_anime.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_depth.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_depth_anime.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_openpose_anime.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_openpose_anime_v2.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/kohya_controllllite_xl_scribble_anime.safetensors -P ./custom_nodes/ControlNet-LLLite-ComfyUI/models
# RUN wget -c https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/control-lora-openposeXL2-rank256.safetensors -P ./models/controlnet/

# RUN echo "Downloading LLavacheckpoints..."  

# RUN wget -c https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q5_K_M.gguf -P ./models/LLavacheckpoints/ 
# #RUN wget -c https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q4_K_M.gguf -P ./models/LLavacheckpoints/ 
# #RUN wget -c https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf -P ./models/LLavacheckpoints/ 
# #RUN wget -c https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q3_K_XS.gguf -P ./models/LLavacheckpoints/ 
# #RUN wget -c https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/llava-v1.6-mistral-7b.Q3_K_M.gguf -P ./models/LLavacheckpoints/ 
# #RUN wget -c https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF/resolve/main/qwen1_5-14b-chat-q4_k_m.gguf  -P ./models/LLavacheckpoints/ 
# #RUN wget -c https://huggingface.co/Qwen/Qwen1.5-14B-Chat-GGUF/resolve/main/qwen1_5-14b-chat-q6_k.gguf -P ./models/LLavacheckpoints/
# RUN wget -c https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf -P ./models/LLavacheckpoints/


# RUN echo "Downloading IPAdapter Plus..."  
# RUN mkdir -p ./models/ipadapter
# RUN wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors -P ./models/ipadapter
# RUN wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors -P ./models/ipadapter 
# RUN wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors -P ./models/ipadapter
# RUN wget -c https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors -P ./models/ipadapter
# RUN wget -c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin -P ./models/ipadapter 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/ip-adapter_sdxl.safetensors -P ./models/ipadapter
# RUN wget -c https://huggingface.co/ostris/ip-composition-adapter/resolve/main/ip_plus_composition_sdxl.safetensors -P ./models/ipadapter

# RUN echo "Downloading ClipVision..."  

# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors -P ./models/clip_vision/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/sd15_model.safetensors -P ./models/clip_vision/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/sdxl_model.safetensors -P ./models/clip_vision/ 
# RUN wget -c https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sd15.bin -P ./models/clip_vision/ 
# #RUN wget -c https://huggingface.co/TryOnVirtual/Ip-Adapter-Cloth/resolve/main/model.safetensors -P ./models/clip_vision/ 

# RUN echo "Downloading ClipVision..."  

# RUN wget -c https://huggingface.co/ckpt/stable-diffusion-3-medium/resolve/main/text_encoders/clip_g.safetensors -P ./models/clip/ 
# RUN wget -c https://huggingface.co/ckpt/stable-diffusion-3-medium/resolve/main/text_encoders/clip_l.safetensors -P ./models/clip/ 
# RUN wget -c https://huggingface.co/ckpt/stable-diffusion-3-medium/resolve/main/text_encoders/t5xxl_fp16.safetensors -P ./models/clip/ 

# RUN echo "Downloading Lora..."  

# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/lcm-lora-sdv1-5.safetensors -P ./models/loras/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/lcm-lora-sdxl.safetensors -P ./models/loras/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/3DMM_V12_sd15.safetensors -P ./models/loras/  
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/add_detail.safetensors -P ./models/loras/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/anime_lineart_lora.safetensors -P ./models/loras/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/epi_noiseoffset2_sd15.safetensors -P ./models/loras/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/game_bottle_lora.safetensors -P ./models/loras/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/game_sword_lora.safetensors -P ./models/loras/ 

# RUN echo "Downloading Motion Lora..."  

# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingAnticlockwise.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_RollingClockwise.ckpt  -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltDown.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_TiltUp.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomIn.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_ZoomOut.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/StopMotionAnimation.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 
# RUN wget -c https://huggingface.co/ArtGAN/Controlnet/resolve/main/shatterAnimatediff_v10.safetensors -P ./custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/ 

# RUN echo "Downloading SUPIR..."  

# #RUN wget -c https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0Q.ckpt -P ./models/checkpoints/ 
# #RUN wget -c https://huggingface.co/camenduru/SUPIR/resolve/main/SUPIR-v0F.ckpt -P ./models/checkpoints/ 

# RUN echo "Downloading BiRefNet..."  
# RUN cd models && git clone https://huggingface.co/ViperYX/BiRefNet 

RUN echo "Done"


CMD python main.py --listen 0.0.0.0 --port 7860 --multi-user ${USE_PERSISTENT_DATA:+--output-directory=/data/}
