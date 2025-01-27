from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.filter
def get_from_dict(key, dictionary):
    return dictionary.get(key) 