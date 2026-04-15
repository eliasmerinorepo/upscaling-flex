from __future__ import annotations
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from upscaler_core import (
    MODEL_CONFIGS,
    ensure_dirs,
    get_default_denoise,
    get_model_description,
    model_supports_denoise_mix,
    upscale_array,
)


APP_STYLES = """
QMainWindow {
  background: #f4ede2;
}
QFrame#Card {
  background: #fff8ee;
  border: 1px solid #dccab7;
  border-radius: 22px;
}
QLabel#HeroTitle {
  color: #231811;
  font-size: 34px;
  font-weight: 700;
}
QLabel#HeroBody, QLabel#Body {
  color: #6f6258;
  font-size: 13px;
}
QLabel#SectionTitle {
  color: #231811;
  font-size: 17px;
  font-weight: 700;
}
QPushButton {
  background: #c45c3b;
  color: white;
  border: none;
  border-radius: 14px;
  padding: 12px 16px;
  font-size: 13px;
  font-weight: 700;
}
QPushButton:hover {
  background: #a84f31;
}
QPushButton:disabled {
  background: #d9c5b7;
  color: #f7f1eb;
}
QComboBox, QTextEdit {
  background: #fffdf9;
  color: #231811;
  border: 1px solid #dccab7;
  border-radius: 12px;
  padding: 10px 12px;
  font-size: 13px;
}
QSlider::groove:horizontal {
  height: 8px;
  background: #ead8c5;
  border-radius: 4px;
}
QSlider::handle:horizontal {
  background: #c45c3b;
  width: 18px;
  margin: -6px 0;
  border-radius: 9px;
}
QLabel#PreviewTitle {
  color: #231811;
  font-size: 18px;
  font-weight: 700;
}
QLabel#PreviewPane {
  background: #efe2d2;
  border-radius: 18px;
  border: 1px solid #e5d4c3;
}
QStatusBar {
  color: #6f6258;
}
"""


class UpscaleWorker(QObject):
    finished = Signal(object, str, str)
    failed = Signal(str)

    def __init__(self, source_array: np.ndarray, model_name: str, scale: float, denoise: float) -> None:
        super().__init__()
        self.source_array = source_array
        self.model_name = model_name
        self.scale = scale
        self.denoise = denoise

    def run(self) -> None:
        try:
            result_array, details, output_path = upscale_array(
                self.source_array,
                self.model_name,
                self.scale,
                self.denoise,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return

        self.finished.emit(result_array, details, str(output_path))


class ImagePane(QFrame):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setObjectName("Card")
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("PreviewTitle")
        layout.addWidget(title_label)

        self.image_label = QLabel("Sin imagen")
        self.image_label.setObjectName("PreviewPane")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 320)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)

    def set_pil_image(self, image: Image.Image | None) -> None:
        if image is None:
            self._pixmap = None
            self.image_label.setText("Sin imagen")
            self.image_label.setPixmap(QPixmap())
            return

        qim = pil_to_qimage(image)
        self._pixmap = QPixmap.fromImage(qim)
        self._refresh_pixmap()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap is None:
            return
        target_size = self.image_label.size() - QSize(24, 24)
        scaled = self._pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.setText("")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UPSCALING FLEX")
        self.resize(1360, 880)
        self.setMinimumSize(1160, 760)

        self.source_path: Path | None = None
        self.source_image: Image.Image | None = None
        self.source_array: np.ndarray | None = None
        self.result_image: Image.Image | None = None
        self.result_path: Path | None = None

        self.worker_thread: QThread | None = None
        self.worker: UpscaleWorker | None = None

        self._build_ui()
        self._apply_model_state()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(18)

        hero = QWidget()
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(0, 0, 0, 0)
        hero_layout.setSpacing(20)

        cat_label = QLabel()
        cat_pixmap = QPixmap(str(Path(__file__).parent / "cat.jpg"))
        if not cat_pixmap.isNull():
            cat_pixmap = cat_pixmap.scaled(
                QSize(220, 220),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        cat_label.setPixmap(cat_pixmap)
        cat_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        cat_label.setFixedSize(220, 220)
        hero_layout.addWidget(cat_label)

        hero_text = QWidget()
        hero_text_layout = QVBoxLayout(hero_text)
        hero_text_layout.setContentsMargins(0, 0, 0, 0)
        hero_text_layout.setSpacing(8)
        hero_text_layout.setAlignment(Qt.AlignVCenter)

        title = QLabel("UPSCALING FLEX")
        title.setObjectName("HeroTitle")
        hero_text_layout.addWidget(title)

        body = QLabel("a cool AI tool for the wonderful students of UDIT.")
        body.setObjectName("HeroBody")
        body.setWordWrap(True)
        hero_text_layout.addWidget(body)
        hero_layout.addWidget(hero_text, 1)

        root.addWidget(hero)

        content = QGridLayout()
        content.setHorizontalSpacing(18)
        content.setVerticalSpacing(18)
        content.setColumnStretch(0, 0)
        content.setColumnStretch(1, 1)
        content.setColumnStretch(2, 1)
        root.addLayout(content, 1)

        controls = QFrame()
        controls.setObjectName("Card")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(14)
        controls.setFixedWidth(360)

        controls_layout.addWidget(section_label("Controles"))
        controls_layout.addWidget(body_label("Carga una imagen, ajusta el modelo y lanza el upscale."))

        self.open_button = QPushButton("Abrir Imagen")
        self.open_button.clicked.connect(self.open_image)
        controls_layout.addWidget(self.open_button)

        self.path_label = body_label("Ningún archivo seleccionado")
        self.path_label.setWordWrap(True)
        controls_layout.addWidget(self.path_label)

        controls_layout.addWidget(section_label("Modelo"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(MODEL_CONFIGS.keys()))
        self.model_combo.currentTextChanged.connect(self._on_model_change)
        controls_layout.addWidget(self.model_combo)

        self.description_label = body_label("")
        self.description_label.setWordWrap(True)
        controls_layout.addWidget(self.description_label)

        controls_layout.addWidget(section_label("Escala de salida"))
        scale_row = QHBoxLayout()
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(2, 8)
        self.scale_slider.setSingleStep(1)
        self.scale_slider.setValue(8)
        self.scale_slider.valueChanged.connect(self._on_scale_change)
        scale_row.addWidget(self.scale_slider, 1)
        self.scale_value_label = body_label("")
        scale_row.addWidget(self.scale_value_label)
        controls_layout.addLayout(scale_row)

        controls_layout.addWidget(section_label("Noise reduction mix"))
        denoise_row = QHBoxLayout()
        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setRange(0, 100)
        self.denoise_slider.valueChanged.connect(self._on_denoise_change)
        denoise_row.addWidget(self.denoise_slider, 1)
        self.denoise_value_label = body_label("")
        denoise_row.addWidget(self.denoise_value_label)
        controls_layout.addLayout(denoise_row)

        self.denoise_hint_label = body_label("")
        self.denoise_hint_label.setWordWrap(True)
        controls_layout.addWidget(self.denoise_hint_label)

        self.run_button = QPushButton("Hacer Upscale")
        self.run_button.clicked.connect(self.run_upscale)
        controls_layout.addWidget(self.run_button)

        self.save_button = QPushButton("Guardar Copia Como...")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_copy)
        controls_layout.addWidget(self.save_button)
        controls_layout.addStretch(1)

        content.addWidget(controls, 0, 0, 1, 1)

        self.original_pane = ImagePane("Original")
        self.result_pane = ImagePane("Resultado")
        content.addWidget(self.original_pane, 0, 1)
        content.addWidget(self.result_pane, 0, 2)

        self.details_box = QTextEdit()
        self.details_box.setReadOnly(True)
        self.details_box.setMinimumHeight(140)
        self.details_box.setPlainText("Aún no has procesado ninguna imagen.")
        root.addWidget(self.details_box)

        self.setCentralWidget(central)

        status = QStatusBar()
        status.showMessage("Listo para cargar una imagen.")
        self.setStatusBar(status)

        open_action = QAction("Abrir Imagen", self)
        open_action.triggered.connect(self.open_image)
        self.addAction(open_action)

    def _on_model_change(self, *_: object) -> None:
        self._apply_model_state()

    def _on_scale_change(self) -> None:
        self.scale_value_label.setText(f"x{self.output_scale():.1f}")

    def _on_denoise_change(self) -> None:
        self.denoise_value_label.setText(f"{self.denoise_strength():.2f}")

    def _apply_model_state(self) -> None:
        model_name = self.model_combo.currentText()
        self.description_label.setText(get_model_description(model_name))
        self.denoise_slider.setValue(int(get_default_denoise(model_name) * 100))
        enabled = model_supports_denoise_mix(model_name)
        self.denoise_slider.setEnabled(enabled)
        self.denoise_hint_label.setText(
            "Activa mezcla entre detalle y reducción de ruido."
            if enabled
            else "Solo se usa en `realesr-general-x4v3`."
        )
        self._on_scale_change()
        self._on_denoise_change()

    def output_scale(self) -> float:
        return self.scale_slider.value() / 2

    def denoise_strength(self) -> float:
        return self.denoise_slider.value() / 100

    def open_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecciona una imagen",
            "",
            "Imágenes (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;Todos los archivos (*)",
        )
        if not file_path:
            return

        path = Path(file_path)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "No pude abrir la imagen", str(exc))
            return

        self.source_path = path
        self.source_image = image
        self.source_array = np.array(image)
        self.result_image = None
        self.result_path = None

        self.path_label.setText(str(path))
        self.original_pane.set_pil_image(image)
        self.result_pane.set_pil_image(None)
        self.details_box.setPlainText(
            f"Original: {image.width} x {image.height} px\nModelo actual: {self.model_combo.currentText()}"
        )
        self.save_button.setEnabled(False)
        self.statusBar().showMessage("Imagen cargada. Ya puedes lanzar el upscale.")

    def run_upscale(self) -> None:
        if self.source_array is None:
            QMessageBox.information(self, "Falta imagen", "Primero abre una imagen.")
            return
        if self.worker_thread and self.worker_thread.isRunning():
            return

        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.details_box.setPlainText("Trabajando...\nLa primera vez puede tardar porque descarga pesos.")
        self.statusBar().showMessage("Procesando imagen...")

        self.worker_thread = QThread(self)
        self.worker = UpscaleWorker(
            self.source_array,
            self.model_combo.currentText(),
            self.output_scale(),
            self.denoise_strength(),
        )
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._handle_success)
        self.worker.failed.connect(self._handle_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_worker)
        self.worker_thread.start()

    def _handle_success(self, result_array: object, details: str, output_path: str) -> None:
        image = Image.fromarray(result_array)
        self.result_image = image
        self.result_path = Path(output_path)
        self.result_pane.set_pil_image(image)
        self.details_box.setPlainText(details)
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.statusBar().showMessage(f"Upscale completado. Guardado en {self.result_path.name}")

    def _handle_error(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.details_box.setPlainText("Error durante el procesado.")
        self.statusBar().showMessage("Ha fallado el upscale.")
        QMessageBox.critical(self, "No pude procesar la imagen", message)

    def _cleanup_worker(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.worker_thread is not None:
            self.worker_thread.deleteLater()
            self.worker_thread = None

    def save_copy(self) -> None:
        if self.result_path is None or self.result_image is None:
            return

        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar copia",
            self.result_path.name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WebP (*.webp)",
        )
        if not destination:
            return

        dest_path = Path(destination)
        if dest_path.resolve() == self.result_path.resolve():
            self.statusBar().showMessage(f"El archivo ya existe en {dest_path.name}")
            return

        image_to_save = self.result_image
        if dest_path.suffix.lower() in {".jpg", ".jpeg"}:
            image_to_save = self.result_image.convert("RGB")
        image_to_save.save(dest_path)
        self.statusBar().showMessage(f"Copia guardada en {dest_path.name}")


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SectionTitle")
    return label


def body_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Body")
    return label


def pil_to_qimage(image: Image.Image) -> QImage:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    return QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888).copy()


def main() -> None:
    ensure_dirs()
    app = QApplication([])
    app.setStyleSheet(APP_STYLES)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
