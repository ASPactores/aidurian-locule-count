
# RUN mkdir /app
# ADD . /app
# WORKDIR /app
# COPY ./requirements.txt .

FROM ultralytics/ultralytics:latest
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8888


# COPY ./src ./src


# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]