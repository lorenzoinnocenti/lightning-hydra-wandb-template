# Python ML template
A simple template to bootstrap Python machine learning projects, maintaining a standard structure.
It should be slightly faster than starting from scratch each time.

## Getting Started

To use this template, follow these steps:

1. Click the "Use this template" button at the top of the repository and follow the procedure.

2. Clone your new repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

3. Navigate to the project directory:

   ```bash
   cd your-repo
   ```

4. The template provides a simple "self-destructing" initialization script, `init.py`, that automatically sets up the package name, author, and other metadata. Run it with:

   ```bash
   uv run python init.py
   ```

5. Install the project and its dependencies:

   ```bash
   # Bare minimum
   uv sync
   # With dev, docs, and test extras
   uv sync --extra dev --extra docs --extra test
   ```

6. You're good to go! Of course, you can further customize it to your liking.

> **Note**
>
> The `init.py` script is self-contained and will delete itself once the procedure is completed. It is absolutely safe to delete if you prefer to edit the files manually.

## Extra goodies

If you are using VS Code as your editor of choice, you can use the following
snippet in your `settings.json` file to format and sort imports on save.

```json
{
    "[python]": {
        "diffEditor.ignoreTrimWhitespace": false,
        "editor.wordBasedSuggestions": "off",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        },
        "editor.defaultFormatter": "charliermarsh.ruff",
    },
}
```
Of course, this is completely optional.
