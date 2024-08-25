# Text Mining Application

## Descripción

Esta aplicación de minería de texto permite ingresar un texto en una caja de texto y obtener resultados de análisis, como la identificación de entidades clave, análisis de sentimientos y resumen automático. La aplicación está desarrollada en Flask.

## Instalación

1. **Clona el repositorio**:
   ```
   git clone https://github.com/SDDrouet/Text-Mining-Application.git
   ```

2. Navega al directorio del proyecto:
    ```
    cd Text-Mining-Application
    ```
3. Crea un entorno virtual:
    ```
    python -m venv venv
    ```
4. Activa el entorno virtual:

    En Windows:

    ```
    venv\Scripts\activate
    ```
    En macOS y Linux:

    ```
    source venv/bin/activate
    ```

5. Instala las dependencias:

    ```
    pip install -r requirements.txt
    ```
    Ejecuta la aplicación:
    ```
    python app.py
    ```

    Accede a la aplicación: Abre tu navegador web y visita http://127.0.0.1:5000/ para usar la aplicación.

## Uso
1. Ingresa el texto que deseas analizar en la caja de texto proporcionada.
2. Presiona el botón para obtener los resultados del análisis.
3. Revisa los resultados mostrados en la página.
## Requisitos
- Python 3.x
- Flask
- Otras dependencias listadas en requirements.txt