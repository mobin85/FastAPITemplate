# FastAPI Template & Manager CLI

A powerful CLI tool designed to scaffold and manage FastAPI projects using modern best practices with **FastAPI**, **SQLModel**, **Alembic**, and **uv**. It provides a structure similar to Django but tailored for the async Python ecosystem.

## Installation

You can install the tool directly from the source:

```bash
git clone https://github.com/mobin85/FastAPITemplate
cd FastAPITemplate
pip install -e .
```

Once installed, the `fastapi-manage` command will be available globally (in your active environment).

## Usage

### 1. Start a New Project
Create a complete, ready-to-use FastAPI project structure with one command. This sets up the directory layout, configuration files (`pyproject.toml`, `alembic.ini`, `.env`), and core application files.

```bash
fastapi-manage startproject <project_name>
```
*Example:* `fastapi-manage startproject mybackend`

This will:
- Initialize a new project using `uv`.
- Install necessary dependencies (FastAPI, SQLModel, Uvicorn, etc.).
- Set up an async database engine (SQLite by default).
- Configure Alembic for async migrations.

### 2. Run Development Server
Navigate into your project directory and start the server. This command wraps `uvicorn`.

```bash
cd <project_name>
fastapi-manage run --reload
```
You can pass any standard Uvicorn arguments to this command.

### 3. Create a New App (Module)
Generate a new modular app with a standard structure (Service, Repository, Models, Views) inside your `app/` directory.

```bash
fastapi-manage startapp <app_name>
```
*Example:* `fastapi-manage startapp users`

Options:
- `--with-enums`: Create an `enums.py` file.
- `--force`: Overwrite existing files if they exist.

### 4. Database Migrations
Manage your database schema changes easily with Alembic wrappers.

**Generate a new migration (autogenerate):**
```bash
fastapi-manage makemigrations -m "description of changes"
```

**Apply migrations (upgrade to head):**
```bash
fastapi-manage migrate
```

## Project Structure
The generated project follows a modular architecture:

```
myproject/
├── app/
│   ├── base/           # Base classes (Service, Repository, Engine, Model)
│   ├── brand/          # Example app module
│   │   ├── models.py
│   │   ├── services.py
│   │   └── ...
│   ├── core/           # Core configurations (Settings, Middleware)
│   ├── main.py         # App entrypoint
│   └── routers.py      # Main router inclusion
├── alembic/            # Migration scripts
└── pyproject.toml      # Dependencies & config
```
