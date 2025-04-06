from django import forms
from .models import userProfile

class userProfileForm(forms.ModelForm):
    class Meta:
        model = userProfile
        fields = ['org_name', 'org_address', 'org_phone_number', 'org_email']
 