# Upscaling IA nativa

App nativa de escritorio para hacer upscaling de imagenes con modelos open source de la familia Real-ESRGAN.

## Modelos recomendados

- `RealESRGAN x4 Plus`: la mejor opcion general para fotos.
- `RealESRNet x4 Plus`: mas conservador, util cuando la imagen ya esta bastante limpia.
- `RealESRGAN x4 Plus Anime`: ideal para ilustracion y anime.
- `realesr-general-x4v3`: rapido y muy buen punto de partida para imagenes pequenas o comprimidas.

## Requisitos

- Python 3.11 recomendado
- Conexion a internet en el primer arranque para descargar los pesos del modelo
- Python 3.14 da problemas de compatibilidad con `basicsr`

## Instalacion

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar

```bash
.venv/bin/python app.py
```

## App de macOS

Ya he generado el bundle nativo en:

- `dist/Upscaling IA.app`

Puedes abrirlo con doble clic desde Finder o con:

```bash
open "dist/Upscaling IA.app"
```

Si quieres volver a compilar la app:

```bash
./build_macos_app.sh
```

La app abre una ventana nativa con:

- selector de imagen
- selector de modelo
- vista previa original y resultado
- guardado automatico en `outputs/`
- opcion de guardar una copia donde quieras

La app crea:

- `weights/` para los modelos descargados
- `outputs/` para las imagenes procesadas

## Notas

- La primera ejecucion tarda mas porque descarga pesos.
- En CPU funciona, pero puede ser lenta con imagenes grandes.
- El bundle de macOS ocupa bastante espacio porque incluye Qt, PyTorch y dependencias cientificas.
