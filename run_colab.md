# Run the app in Colab / Kaggle (GPU required)

Cell 1:
```python
!pip install -q -r requirements.txt
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
```

Cell 2:
```python
import subprocess, time
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(10)
!./cloudflared tunnel --url http://localhost:8501
```

Open the `trycloudflare.com` link it prints. Record your demo there.
