# HW1
## PyQt5
### How to install and run designer tool?
* install
    ```sh
    pip install pyqt5
    pip install PyQt5-sip
    pip install pyqt5-tools
    ```
* running
    * find designer.exe and run
    * select Main Window to create new UI
    * using IDE to design your UI
    * save
    * transfer ui file into python file
        ```sh
        pyuic5 {PyQt_UI_FILE_NAME}.ui -o {PyQt_PY_FILE_NAME}.py
        ```
    * create main.py to show the ui
    * create executable file
        ```sh
        pyinstaller -F ./main.py
        ```