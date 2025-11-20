# LivePortrait Server Remodel
Forked from https://github.com/KlingTeam/LivePortrait

This is a server intended to take a source image and a list of facial landmarks, and output the modified source image in a REST API call.

An example code is in `test_edit_face_image.py`

The general steps are:
1) Register connection via `/register-connection`. This instantiates a pipeline object and will return a UUID that accesses this pipeline object.
2) Process the source image via `/process-source`. Sending a UUID and base64-encoded image will pre-process the source image for the respective pipeline object
3) Sending a driving base64-encoded pickle data of the landmarks to `/edit-face-landmarks` will return the modified image in base64-encoded format.

The JSON format for each calls are shown in the example code and in `server.py`
