from django.db import models
from django.contrib.auth.models import User


class userProfile(models.Model):
    user = models.OneToOneField(User, null=True, on_delete=models.CASCADE)
    tnc_flag = models.BooleanField(default=False)
    org_name = models.CharField(max_length=50)
    org_address = models.CharField(null=True, blank=True, max_length=500)
    org_phone_number = models.CharField(max_length=15, null=True, blank=True)
    org_email = models.EmailField(blank=True, null=True)

    def __str__(self):
        return str(self.user)
    class Meta:
        db_table = 'userProfile'