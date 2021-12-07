# bda-fullend-webapp
### Environment
Requires Docker installed

Make sure Docker have 5GB memory

### Download model weight
Since the weight is too big, we need to download and put into model_weights/

MFAS weight download: https://drive.google.com/file/d/1R_6OymhbK4sQv6p-o0PMfGFrpfwaE1xa/view?usp=sharing

CLIP weight download: https://drive.google.com/file/d/1OgNlQQLiso0F1P03m4CG4Kb6B022_ItU/view?usp=sharing

### **Reminder**
:exclamation: Be aware that `.env` should be up-to-date with your twitter developer settings.

### Build docker image and run
```
    > docker-compose build --no-cache
    > docker-compose up
```
You should be able to check backend with URL `http://localhost:5000/`

You should be able to check frontend with URL `http://localhost:8080/`

