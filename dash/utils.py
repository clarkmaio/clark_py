import dash




def build_options(values) -> dict:
    '''
    Return list of dictionaries with keys: label, value.
    usefull to feed dropdown object
    '''
    return [{'label': x, 'value': x} for x in values]

def return_width_height(width_input, height_input, width_default, height_default):
    '''Return width height according to default and input values'''

    if width_input is not None:
        width_output = width_input
    else:
        width_output = width_default

    if height_input is not None:
        height_output = height_input
    else:
        height_output = height_default

    return width_output, height_output