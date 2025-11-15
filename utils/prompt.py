from jinja2 import Template


def compile_prompt(prompt_template_path: str, template_params: dict = None) -> str:
    if template_params is None:
        template_params = {}

    with open(prompt_template_path, encoding="utf-8") as file:
        prompt_template = file.read()

    return Template(prompt_template).render(**template_params)
