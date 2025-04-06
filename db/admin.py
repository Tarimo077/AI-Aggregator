from django.contrib import admin
from .models import userProfile

# Register your models here.
@admin.register(userProfile)
class userProfileAdmin(admin.ModelAdmin):
    list_display = ('tnc_flag', 'org_name')
