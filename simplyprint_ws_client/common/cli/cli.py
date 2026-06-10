__all__ = ["ClientCli", "CommandBag"]

from typing import Any, Dict, Optional, Union, get_args, get_origin, Callable

import click

from typing import TYPE_CHECKING

from simplyprint_ws_client.common.debug.connectivity import ConnectivityReport
from simplyprint_ws_client.const import APP_DIRS

if TYPE_CHECKING:
    from simplyprint_ws_client.core.app import ClientApp
    from simplyprint_ws_client.core.config import PrinterConfig


class CommandBag:
    commands: Dict[str, click.Command]

    def list_commands(self, ctx):
        return sorted(self.commands.keys())

    def get_command(self, ctx, name):
        return self.commands.get(name)

    def add_command(self, command: click.Command):
        self.commands[command.name] = command


class ClientCliConfigManager(CommandBag, click.Group):
    """Commands to interact with the configuration manager."""

    def __init__(self, app: "ClientApp") -> None:
        super().__init__(name="config", help="Configuration manager commands")
        self.app = app
        self.commands = {}
        self.add_command(
            click.Command(
                "list",
                callback=self.list_configs,
                params=[self.client_type_option()],
            )
        )
        self.add_command(
            click.Command(
                "edit",
                callback=self.edit_config,
                params=[click.Argument(["index"], type=int), self.client_type_option()],
            )
        )
        self.add_command(
            click.Command(
                "add", callback=self.add_config, params=[self.client_type_option()]
            )
        )
        self.add_command(
            click.Command(
                "new",
                callback=self.add_new_config,
                help="Add a new configuration",
                params=[self.client_type_option()],
            )
        )
        self.add_command(
            click.Command(
                "remove",
                callback=self.remove_config,
                params=[click.Argument(["index"], type=int), self.client_type_option()],
            )
        )

    @staticmethod
    def client_type_option():
        return click.Option(
            ["--type", "client_type"],
            required=False,
            help="Client type to use when multiple client specs are configured.",
        )

    def config_manager(self, client_type: Optional[str] = None):
        if hasattr(self.app, "get_config_manager"):
            return self.app.get_config_manager(client_key=client_type)

        return self.app.config_manager

    def list_configs(self, client_type: Optional[str] = None):
        configs = self.config_manager(client_type).get_all()

        for index, config in enumerate(configs):
            click.echo(f"{index}: {config}")

    def get_config_default(self, field: str, current_value: Any) -> Any:
        if current_value is None and field in ["id", "token"]:
            return 0 if field == "id" else "0"

        if current_value is not None:
            return current_value

        return ""

    def prompt_and_update_config(self, config: "PrinterConfig"):
        click.echo(f"Editing configuration {config}. Leave blank to keep current value")
        fields = list(sorted(config.as_dict()))
        annotations = {}

        for cls in reversed(config.__class__.mro()):
            annotations.update(getattr(cls, "__annotations__", {}))

        for field in fields:
            # Ask user for field value
            field_type = annotations[field] if field in annotations else str

            if get_origin(field_type) is Union:
                field_type = get_args(field_type)[0]

            field_default = self.get_config_default(
                field, getattr(config, field) if hasattr(config, field) else None
            )

            value = click.prompt(
                f"Input {field}",
                default=field_default,
                show_default=field_default != "",
                type=field_type,
            )

            if value is None or value == "":
                continue

            setattr(config, field, value)

    def get_config_by_index(
        self, index: int, client_type: Optional[str] = None
    ) -> Optional["PrinterConfig"]:
        configs = self.config_manager(client_type).get_all()
        if 0 <= index < len(configs):
            return configs[index]
        return None

    def edit_config(self, index: int, client_type: Optional[str] = None):
        manager = self.config_manager(client_type)
        config = self.get_config_by_index(index, client_type)

        if config:
            self.prompt_and_update_config(config)
            manager.persist(config)
            manager.flush()
            click.echo("Configuration updated.")
        else:
            click.echo("Configuration not found.")

    def add_config(self, client_type: Optional[str] = None):
        manager = self.config_manager(client_type)
        config = manager.config_t.get_blank()
        self.prompt_and_update_config(config)
        manager.persist(config)
        manager.flush()
        click.echo("Configuration added.")

    def add_new_config(self, client_type: Optional[str] = None):
        manager = self.config_manager(client_type)
        config = manager.config_t.get_new()
        manager.persist(config)
        manager.flush()
        click.echo("Configuration added.")

    def remove_config(self, index: int, client_type: Optional[str] = None):
        manager = self.config_manager(client_type)
        config = self.get_config_by_index(index, client_type)

        if config:
            manager.remove(config)
            manager.flush()

            click.echo("Configuration removed.")
        else:
            click.echo("Configuration not found.")


class ClientCliDebugConnectivity(CommandBag, click.Group):
    """Commands to debug connectivity."""

    def __init__(self, app: "ClientApp") -> None:
        super().__init__(name="connectivity", help="Debug connectivity commands")
        self.app = app
        self.commands = {}
        self.add_command(click.Command("test", callback=self.test))
        self.add_command(click.Command("status", callback=self.list_previous_reports))
        self.add_command(click.Command("clean", callback=self.delete_all_reports))

    @staticmethod
    def test():
        path = APP_DIRS.user_log_path / "connectivity_reports"

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        r = ConnectivityReport.generate_default()
        p = r.store_in_path(path)
        print("Connectivity test suite complete. Report saved to:", p)

    @staticmethod
    def list_previous_reports():
        path = APP_DIRS.user_log_path / "connectivity_reports"

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        list_of_reports = ConnectivityReport.read_previous_reports(path)

        print(f"Found {len(list_of_reports)} reports in {path}.")

        for report in list_of_reports:
            print(f"* Report generated on {report.timestamp}:")
            print("  - " + "\n  - ".join(report.summary().split("\n")).strip("- "))

    @staticmethod
    def delete_all_reports():
        path = APP_DIRS.user_log_path / "connectivity_reports"

        if not path.exists():
            print("Nothing to delete")

        files = path.rglob("connectivity_report_*.json")

        for file in files:
            print(f"Deleting {file}")
            file.unlink()


class ClientCli(CommandBag, click.MultiCommand):
    app: "ClientApp"
    commands: Dict[str, click.Command]
    _client_runner: Optional[Callable[[], None]] = None

    def __init__(self, app: Optional[ClientApp] = None) -> None:
        super().__init__(name="simplyprint", help="SimplyPrint client CLI")
        self.app = app
        self.commands = {}

        # Register commands
        self.add_command(ClientCliConfigManager(self.app))
        self.add_command(ClientCliDebugConnectivity(self.app))
        self.add_command(
            click.Command("start", callback=self.start_client, help="Start the client")
        )

    @property
    def start_client(self):
        if self._client_runner is None:
            return lambda: self.app.run_blocking()

        return self._client_runner

    @start_client.setter
    def start_client(self, value):
        self._client_runner = value
