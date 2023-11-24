# Activating Pre-commit Hook
This repository includes a pre-commit hook to prevent large file commits. The pre-commit hook checks the size of files before allowing a commit to proceed. If a file exceeds the specified limit, the commit will be aborted.

## Prerequisites
- [Git](https://git-scm.com/) must be installed on your machine.
## Setup Instructions
1. **Clone the Repository:**

    ```bash
    git clone git@github.com:tradingstrategy-ai/trade-executor.git
    ```
2. **Navigate to the Repository:**

    ```bash
    cd trade-executor
    ```

3. **Copy the Pre-commit Hook:**

    Run script to copy the pre-commit file from this repository to the `.git/hooks/` directory of project and add it to git stage

    ```bash
    bash scripts/set-pre-commit-checkfilesize.sh 
    ```
## Configuration
You can customize the behavior of the pre-commit hook by adjusting the MAX_FILE_SIZE variable in the pre-commit script. This variable represents the maximum allowed file size in megabytes.

```bash
# Set the maximum allowed file size in megabytes
vim .git/hooks/pre-commit
# Edit MAX_FILE_SIZE
MAX_FILE_SIZE=35
```

Modify the **MAX_FILE_SIZE** value according to project requirements.

## Testing the Pre-commit Hook
To test the pre-commit hook, attempt to commit changes to repository. If a file exceeds the specified size limit, the commit will be aborted with an error message indicating the problematic file.

## Troubleshooting
- Double-check the MAX_FILE_SIZE variable in the pre-commit script to ensure it meets your project's requirements.

