"""
********************************************************************************************************************************************
****************************************************    GRADIO THEMES    *******************************************************************
********************************************************************************************************************************************
This file contains a collection of Gradio themes that can be used to customize the appearance of Gradio interfaces.
"""

import gradio as gr


default_theme = gr.themes.Default()


theme1 = gr.themes.Ocean(
    spacing_size="lg",
)


theme2 = theme = gr.themes.Ocean(
    primary_hue="blue",
    secondary_hue="blue",
    radius_size="md",
    font=['IBM Plex Sans', 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
).set(
    border_color_accent_dark='*primary_600',
    border_color_primary_dark='*primary_600',
    block_shadow='none',
    block_shadow_dark='none',
    checkbox_background_color_selected='*color_accent',
    checkbox_background_color_selected_dark='*color_accent',
    checkbox_label_background_fill_selected='*checkbox_label_background_fill',
    checkbox_label_background_fill_selected_dark='*checkbox_label_background_fill',
    checkbox_label_border_color_selected='*checkbox_label_border_color',
    checkbox_label_border_color_selected_dark='*checkbox_label_border_color',
    checkbox_label_border_width='*input_border_width',
    checkbox_label_text_color_selected='*checkbox_label_text_color',
    checkbox_label_text_color_selected_dark='*checkbox_label_text_color',
    slider_color='*color_accent',
    slider_color_dark='*color_accent',
    button_border_width='*input_border_width',
    button_border_width_dark='*input_border_width',
    button_transform_hover='none',
    button_transition='all 0.2s ease',
    button_primary_background_fill='*primary_500',
    button_primary_background_fill_dark='*primary_600',
    button_primary_background_fill_hover='*primary_600',
    button_primary_background_fill_hover_dark='*primary_700',
    button_primary_text_color='white',
    button_primary_text_color_dark='white',
    button_primary_shadow_hover='*button_primary_shadow',
    button_secondary_background_fill='*neutral_200',
    button_secondary_background_fill_dark='*neutral_600',
    button_secondary_background_fill_hover='*neutral_300',
    button_secondary_background_fill_hover_dark='*neutral_700',
    button_secondary_shadow_hover='*button_secondary_shadow',
    button_secondary_shadow_dark='*button_primary_shadow'
)


theme3 = gr.themes.Ocean().set(
    background_fill_primary='*white',
    background_fill_secondary='*neutral_700',
    background_fill_secondary_dark='*neutral_800',
    border_color_primary_dark='*primary_800',
    color_accent_soft_dark='*primary_700'
)


theme4 =gr.Theme.from_hub('JohnSmith9982/small_and_pretty')


theme5 = gr.Theme.from_hub('d8ahazard/material_design_blue')