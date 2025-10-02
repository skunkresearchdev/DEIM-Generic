**Prompt:**
Design a script that facilitates the creation of a new AI training module. The script should perform the following tasks:

1. **Directory Structure**: 
   - Create a clean directory structure for the new project, ensuring it includes all necessary folders for modules, data, and scripts.
   - Copy required files and folders from the `_old` directory to the new project directory without altering the logic for training. do not edit any files in _old and do not import from old ever!

2. **Module Creation**:
   - In the root of the new project directory, create a new Python script that serves as the entry point for both training and inference.
   - Ensure the new module can be imported easily and allows for training and inference to be executed in fewer than 10 lines of code.

3. **Dependencies**:
   - Utilize the `supervision` package for annotation purposes, ensuring users can easily add annotated files for training.
   - Include instructions within the script for users to run the training and inference processes.

4. **Environment Configuration**:
   - Specify the use of the Python interpreter located at `/home/hidara/miniconda3/envs/deim/bin/python` throughout the project.
   - Implement linting with `ruff` and type checking with `pyright` to ensure code quality.

5. **User Guidance**:
   - Add comments and documentation within the script to guide new users who may be unfamiliar with AI training processes, focusing on ease of use for adding new annotated files for training and transfer learning.

6. **Execution**:
   - Ensure that the script does not run any training code automatically due to the long-running nature of the training sessions on the GPU.

By following these specifications, the resulting script will create a user-friendly environment for new users to engage with AI training and inference, making the process straightforward and accessible. 