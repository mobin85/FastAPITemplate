#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
import textwrap
import typer


app = typer.Typer(help="Project management CLI (Django-like) for FastAPI + SQLModel")

# ----------------------------
# Helpers
# ----------------------------

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_project_name(name: str) -> None:
    # Project folder name can be more permissive, but Python package names should be valid identifiers.
    if not NAME_RE.match(name):
        raise typer.BadParameter(
            "Invalid name. Use letters/numbers/underscore, and start with a letter or underscore."
        )


def ensure_empty_or_force(path: Path, force: bool) -> None:
    if path.exists():
        if any(path.iterdir()) and not force:
            raise typer.BadParameter(
                f"Target directory is not empty: {path} (use --force to overwrite)"
            )


def write_text(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise typer.BadParameter(f"File exists: {path} (use --force to overwrite)")
    path.write_text(content, encoding="utf-8")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None)


# ----------------------------
# Templates
# ----------------------------


def tpl_env_example(project_name: str) -> str:
    return textwrap.dedent(
        f"""\
        # Example environment variables for {project_name}
        APP_NAME="{project_name}"
        DATABASE_URL="sqlite:///./dev.db"
        """
    )


def tpl_readme(project_name: str) -> str:
    return textwrap.dedent(
        f"""\
        # {project_name}

        ## Run (dev)
        ```bash
        uvicorn app.main:get_app --factory --reload
        ```

        ## Create a new app module
        ```bash
        python manage.py startapp user
        ```

        ## Alembic (basic)
        - Config: alembic.ini
        - Env: alembic/env.py
        """
    )


def tpl_scripts_run_dev() -> str:
    return textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -e

        uvicorn app.main:get_app --factory --reload
        """
    )


def tpl_app_init() -> str:
    return ""


def tpl_app_settings() -> str:
    return textwrap.dedent(
        """\
        from pydantic_settings import BaseSettings, SettingsConfigDict


        class Settings(BaseSettings):
            # Comments are intentionally in English.
            app_name: str
            database_url: str

            model_config = SettingsConfigDict(env_file=".env", extra="ignore")


        def get_settings() -> Settings:
            return Settings()
        """
    )


def tpl_app_routers() -> str:
    return textwrap.dedent(
        """\
        from fastapi import APIRouter

        # Import your app routers here. Example:
        # from app.user.views import router as user_router


        # Add them to this list
        routers = [
            # user_router,
        ]


        # Simple health check router
        health_router = APIRouter()
        @health_router.get("/health", tags=["health"])
        def health():
            return {"status": "ok"}


        routers.append(health_router)
        """
    )


def tpl_app_dependencies() -> str:
    return textwrap.dedent(
        """\
        from app.base.engine import get_engine
        from sqlmodel.ext.asyncio.session import AsyncSession


        async def get_session():
            async with AsyncSession(get_engine()) as session:
                try:
                    yield session
                finally:
                    await session.close()
        """
    )


def tpl_app_application() -> str:
    return textwrap.dedent(
        """\
        from contextlib import asynccontextmanager
        from fastapi import FastAPI

        from .settings import get_settings


        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup logic can go here.
            yield
            # Shutdown logic can go here.


        def get_app() -> FastAPI:
            settings = get_settings()
            application = FastAPI(title=settings.app_name, lifespan=lifespan)
            return application
        """
    )


def tpl_app_main() -> str:
    return textwrap.dedent(
        """\
        from app.core.application import get_app
        from .routers import router


        def _include_routers(app):
            app.include_router(router)
            return app


        # Uvicorn factory entrypoint: uvicorn app.main:get_app --factory
        def get_app():
            app = _include_routers(get_app_impl())
            return app


        def get_app_impl():
            return get_app_base()


        def get_app_base():
            # Keep this separated for easier testing/overrides.
            from app.core.application import get_app as _get
            return _get()
        """
    )


# Note: The above can be simplified, but I kept it explicit so you can easily customize.
# If you prefer the simplest version, tell me and I'll trim it.


def tpl_app_main_simpler() -> str:
    return textwrap.dedent(
        """\
        from app.base.engine import get_engine
        from app.core.application import get_app as _get_app
        from app.base.model import Base

        from .routers import routers


        def get_app():
            app = _get_app()

            Base.metadata.create_all(get_engine())

            for router in routers:
                app.include_router(router)

            return app

        """
    )


def tpl_core_middlewares() -> str:
    return textwrap.dedent(
        """\
        from starlette.middleware.base import BaseHTTPMiddleware


        class ExampleMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                # process request
                response = await call_next(request)
                # process response
                return response
        """
    )


def tpl_base_service() -> str:
    return textwrap.dedent(
        """\

        class Service:
            pass
        """
    )


def tpl_base_repository() -> str:
    return textwrap.dedent(
        """\

        class Repository:
            pass
        """
    )


def tpl_base_engine() -> str:
    return textwrap.dedent(
        """\
        from typing import ClassVar, Optional
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.ext.asyncio.engine import AsyncEngine

        from app.core.settings import get_settings


        class Engine:
            _engine: ClassVar[Optional[AsyncEngine]] = None

            @classmethod
            def get_engine(cls):
                if cls._engine is None:
                    settings = get_settings()
                    print("Creating new AsyncEngine...")
                    cls._engine = create_async_engine(
                        settings.database_url,
                        pool_size=50,
                        max_overflow=50,
                        pool_timeout=30,
                        pool_recycle=1800,
                        pool_pre_ping=True,
                        echo_pool=True
                    )
                return cls._engine


        def get_engine():
            return Engine.get_engine()
        """
    )


def tpl_base_model() -> str:
    return textwrap.dedent(
        """\
        from datetime import datetime
        from uuid import UUID, uuid4
        from sqlmodel import SQLModel, Field


        class Base(SQLModel):
            id: UUID = Field(default_factory=uuid4, primary_key=True)
            created_at: datetime = Field(default_factory=datetime.utcnow)
        """
    )


def tpl_alembic_ini(project_name: str) -> str:
    return textwrap.dedent(
        f"""\
        [alembic]
        script_location = alembic
        prepend_sys_path = .
        sqlalchemy.url = sqlite:///./dev.db

        [loggers]
        keys = root,sqlalchemy,alembic

        [handlers]
        keys = console

        [formatters]
        keys = generic

        [logger_root]
        level = WARN
        handlers = console
        qualname =

        [logger_sqlalchemy]
        level = WARN
        handlers =
        qualname = sqlalchemy.engine

        [logger_alembic]
        level = INFO
        handlers =
        qualname = alembic

        [handler_console]
        class = StreamHandler
        args = (sys.stderr,)
        level = NOTSET
        formatter = generic

        [formatter_generic]
        format = %(levelname)-5.5s [%(name)s] %(message)s
        datefmt = %H:%M:%S
        """
    )


def tpl_alembic_env_py() -> str:
    # Minimal env.py wired to SQLModel metadata if you want later.
    return textwrap.dedent(
        """\
        from __future__ import annotations

        from logging.config import fileConfig
        from alembic import context
        from sqlalchemy import engine_from_config, pool

        from app.base.model import Base
        from app.core.settings import get_settings


        config = context.config
        if config.config_file_name is not None:
            fileConfig(config.config_file_name)

        settings = get_settings()
        config.set_main_option("sqlalchemy.url", settings.database_url)

        target_metadata = Base.metadata


        def run_migrations_offline() -> None:
            url = config.get_main_option("sqlalchemy.url")
            context.configure(
                url=url,
                target_metadata=target_metadata,
                literal_binds=True,
                dialect_opts={"paramstyle": "named"},
            )

            with context.begin_transaction():
                context.run_migrations()


        def run_migrations_online() -> None:
            connectable = engine_from_config(
                config.get_section(config.config_ini_section, {}),
                prefix="sqlalchemy.",
                poolclass=pool.NullPool,
            )


            with connectable.connect() as connection:
                context.configure(connection=connection, target_metadata=target_metadata)

                with context.begin_transaction():
                    context.run_migrations()


        if context.is_offline_mode():
            run_migrations_offline()
        else:
            run_migrations_online()
        """
    )


# ----------------------------
# Commands
# ----------------------------


@app.command()
def startproject(
    name: str = typer.Argument(..., help="Project name (folder name)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
):
    """
    Create a new FastAPI project skeleton (using uv).
    """
    validate_project_name(name)

    project_dir = Path(name).resolve()
    # Check if empty or force
    ensure_empty_or_force(project_dir, force=force)

    typer.echo(f"Initializing project {name} with uv...")
    # 1. uv init
    run(["uv", "init", "--no-workspace", name])

    # Remove hello.py generated by uv if it exists
    hello_py = project_dir / "hello.py"
    main_py = project_dir / "main.py"

    if hello_py.exists():
        hello_py.unlink()

    if main_py.exists():
        main_py.unlink()

    # 2. Add dependencies (also creates .venv)
    typer.echo("Installing dependencies...")
    run(
        [
            "uv",
            "add",
            "fastapi",
            "uvicorn[standard]",
            "sqlmodel",
            "alembic",
            "pydantic-settings",
            "aiosqlite",
        ],
        cwd=project_dir,
    )

    # 3. Initialize Alembic
    typer.echo("Initializing Alembic...")
    run(["uv", "run", "alembic", "init", "alembic"], cwd=project_dir)

    # 4. Customizing configs
    # Append tool.uvicorn to pyproject.toml
    pyproject_path = project_dir / "pyproject.toml"
    if pyproject_path.exists():
        with open(pyproject_path, "a", encoding="utf-8") as f:
            f.write("\n[tool.uvicorn]\nfactory = true\n")

    # Create additional directories
    (project_dir / "scripts").mkdir(parents=True, exist_ok=True)
    # Base app structure
    (project_dir / "app" / "base").mkdir(parents=True, exist_ok=True)
    (project_dir / "app" / "core").mkdir(parents=True, exist_ok=True)

    # Write project files
    write_text(project_dir / ".env", tpl_env_example(name), force=force)
    # uv init creates README.md, so we must overwrite it
    write_text(project_dir / "README.md", tpl_readme(name), force=True)

    # Overwrite Alembic files with our templates (alembic init creates these)
    write_text(project_dir / "alembic.ini", tpl_alembic_ini(name), force=True)
    write_text(project_dir / "alembic" / "env.py", tpl_alembic_env_py(), force=True)

    # scripts
    run_dev = project_dir / "scripts" / "run_dev.sh"
    write_text(run_dev, tpl_scripts_run_dev(), force=force)
    try:
        run(["chmod", "+x", str(run_dev)])
    except Exception:
        pass

    # app core
    write_text(project_dir / "app" / "__init__.py", tpl_app_init(), force=force)
    write_text(project_dir / "app" / "routers.py", tpl_app_routers(), force=force)

    # app core (settings, application, middlewares, dependencies)
    write_text(project_dir / "app" / "core" / "__init__.py", "", force=force)
    write_text(
        project_dir / "app" / "core" / "settings.py", tpl_app_settings(), force=force
    )
    write_text(
        project_dir / "app" / "core" / "application.py",
        tpl_app_application(),
        force=force,
    )
    write_text(
        project_dir / "app" / "core" / "middlewares.py",
        tpl_core_middlewares(),
        force=force,
    )
    write_text(
        project_dir / "app" / "core" / "dependencies.py",
        tpl_app_dependencies(),
        force=force,
    )

    write_text(project_dir / "app" / "main.py", tpl_app_main_simpler(), force=force)

    # app base (service, repository, engine, model)
    write_text(project_dir / "app" / "base" / "__init__.py", "", force=force)
    write_text(
        project_dir / "app" / "base" / "service.py", tpl_base_service(), force=force
    )
    write_text(
        project_dir / "app" / "base" / "repository.py",
        tpl_base_repository(),
        force=force,
    )

    write_text(
        project_dir / "app" / "base" / "engine.py", tpl_base_engine(), force=force
    )
    write_text(project_dir / "app" / "base" / "model.py", tpl_base_model(), force=force)

    typer.echo(f"✅ Project created: {project_dir}")
    typer.echo("Next:")
    typer.echo(f"  cd {name}")
    typer.echo("  # .env is already created with default settings.")
    typer.echo("  # Run dev server:")
    typer.echo("  fastapi-manage run ")


@app.command()
def startapp(
    name: str = typer.Argument(..., help="App name (module name), e.g. user"),
    project_dir: Path = typer.Option(
        ".", "--project-dir", help="Path to the project root (where app/ exists)"
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
    with_enums: bool = typer.Option(False, "--with-enums", help="Also create enums.py"),
):
    """
    Create a new app module inside ./app/<name> (Django-like layout).
    """
    validate_project_name(name)

    project_dir = project_dir.resolve()
    app_root = project_dir / "app"
    if not app_root.exists():
        raise typer.BadParameter(f"Could not find app/ directory at: {app_root}")

    target = app_root / name
    target.mkdir(parents=True, exist_ok=True)

    camel_name = "".join(x.title() for x in name.split("_"))

    templates: dict[str, str] = {
        "__init__.py": "",
        "schemas.py": "# Pydantic/SQLModel schemas go here.\n",
        "models.py": "# SQLModel models go here.\n",
        "admin.py": "# Admin hooks (optional) go here.\n",
        "services.py": textwrap.dedent(
            f"""\
            from app.base.service import Service

            class {camel_name}Service(Service):
                pass
            """
        ),
        "repositories.py": textwrap.dedent(
            f"""\
            from app.base.repository import Repository

            class {camel_name}Repository(Repository):
                pass
            """
        ),
        "views.py": "# FastAPI routes/handlers for this app go here.\n",
    }

    if with_enums:
        templates["enums.py"] = "# Enum definitions go here.\n"

    for filename, content in templates.items():
        write_text(target / filename, content, force=force)

    typer.echo(f"✅ App created: {target}")
    typer.echo(
        "Tip: import and include your app router(s) from app/routers.py (or refactor to per-app routers)."
    )


@app.command()
def makemigrations(
    message: str = typer.Option("auto", "--message", "-m", help="Migration message"),
):
    """
    Generate new database migrations (alembic revision --autogenerate).
    """
    typer.echo(f"Creating migration with message: {message}")
    run(["uv", "run", "alembic", "revision", "--autogenerate", "-m", message])


@app.command()
def migrate(
    revision: str = typer.Argument("head", help="Revision to upgrade to"),
):
    """
    Apply database migrations (alembic upgrade).
    """
    typer.echo(f"Applying migrations (upgrade to {revision})...")
    run(["uv", "run", "alembic", "upgrade", revision])


@app.command(
    name="run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_help_option=False,
)
def start_server(ctx: typer.Context):
    """
    Run the dev server (uvicorn). Use --reload or other flags as needed.
    (Passes --help to uvicorn).
    """
    project_dir = Path.cwd().resolve()
    os.chdir(project_dir)

    # Heuristic: if user provides an argument with ":", assume it's the app string (e.g. app.main:app)
    # in that case we don't use the default app.main:get_app --factory
    has_app_string = any((":" in arg and not arg.startswith("-")) for arg in ctx.args)

    cmd = ["uv", "run", "uvicorn"]
    if not has_app_string:
        cmd.append("app.main:get_app")
        cmd.append("--factory")

    if ctx.args:
        cmd.extend(ctx.args)

    typer.echo(f"Running server in {project_dir}...")
    run(cmd)


if __name__ == "__main__":
    app()
