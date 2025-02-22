FROM ultralytics/ultralytics:latest

# Set the working directory inside the container
WORKDIR /app

# Install Times New Roman font (and other Microsoft core fonts) on Ubuntu-based images
# ARG DEBIAN_FRONTEND=noninteractive
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install --no-install-recommends ttf-mscorefonts-installer

# # Set the fontconfig cache to make the newly installed font available
# RUN fc-cache -f -v

# Copy the project files into the container
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose port 8888 for Jupyter or other services running in the container
EXPOSE 8888


