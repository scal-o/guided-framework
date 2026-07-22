import click
from static_assignment import od_convert, sta_run, sta_test


@click.group("sta")
def cli():
    pass


cli.add_command(od_convert.convert_cli)
cli.add_command(sta_run.run_sta)
cli.add_command(sta_test.test_sta)
