# myapp/templatetags/custom_filters.py
from django import template

register = template.Library()

@register.filter
def get_dict_value(dictionary, key):
    """Custom filter to get a value from a dictionary."""
    return dictionary.get(key, None)
