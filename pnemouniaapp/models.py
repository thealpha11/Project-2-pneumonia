from django.db import models 
class radio(models.Model):
    try:
        xray= models.ImageField(upload_to='images/',default="")
    except:
        pass
