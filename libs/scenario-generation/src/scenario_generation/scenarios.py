import click
from scenario_generation import generate_scenarios, conv_to_gpkg


@click.group("scenarios")
def cli():
    pass


cli.add_command(generate_scenarios.generate_scenarios)
cli.add_command(conv_to_gpkg.create_gpkg)
