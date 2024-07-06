# Installation

1. Install Docker Desktop for Mac:
   - Go to the official Docker website: https://www.docker.com/products/docker-desktop
   - Click on "Download for Mac"
   - Once downloaded, double-click the "Docker.dmg" file
   - Drag the Docker icon to your Applications folder

2. Set up Docker Desktop:
   - Open Docker Desktop from your Applications folder
   - You may be prompted to enter your Mac password to allow Docker to make changes
   - Wait for Docker to start (you'll see a whale icon in your menu bar when it's ready)


3. Build the Docker image:
   - In Terminal, ensure you're in the comfyui-docker-mac directory, then run:
     ```
     docker build -t comfyui .
     ```
   - This builds an image named "comfyui" based on the existing Dockerfile
   - Wait for the build process to complete (this may take some time)

4. Run the Docker container:
   - Once the build is finished, run this command:
     ```
     docker run -p 7860:7860 -e USE_PERSISTENT_DATA=1 -v $(pwd)/data:/data comfyui
     ```
   - This starts a container from your image and maps port 7860

5. Access ComfyUI:
   - Open a web browser and go to `http://localhost:7860`
   - You should see the ComfyUI interface running

Additional notes:
- If you encounter any "permission denied" errors, you may need to run the docker commands with `sudo` (e.g., `sudo docker build -t comfyui .`)
- To stop the container, press Ctrl+C in the Terminal window where it's running
- To see a list of running containers, use the command `docker ps`
- To stop a container by its ID, use `docker stop <container_id>`

These updated instructions use the existing Dockerfile in the ComfyUI repository, which should already be optimized for various systems, including Mac M1/M2. If you encounter any issues specific to M1/M2 compatibility, you may need to modify the Dockerfile, but for most users, the existing setup should work well.

