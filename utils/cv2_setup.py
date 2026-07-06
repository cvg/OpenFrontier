import os


def configure_opencv_qt_fonts() -> None:
    if os.environ.get("QT_QPA_FONTDIR"):
        return

    candidate_dirs = [
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
        "/usr/share/fonts/truetype",
        "/usr/share/fonts",
    ]

    for font_dir in candidate_dirs:
        if os.path.isdir(font_dir):
            os.environ["QT_QPA_FONTDIR"] = font_dir
            return
