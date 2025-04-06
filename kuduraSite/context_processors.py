from db.models import userProfile

def user_profile_context(request):
    profile = None
    if request.user.is_authenticated:
        try:
            profile = userProfile.objects.get(user=request.user)
        except userProfile.DoesNotExist:
            profile = None
    return {'profile': profile}
