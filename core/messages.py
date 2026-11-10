import streamlit as st

class Message:
    @staticmethod
    def whatsNew(language="RU"):
        if language == "RU":
            st.markdown("""
                ### **Примеры изображений**
                | Действие | Описание    |
                |----------|-------------|
                | ▶️ **Cегментация** | Нажмите кнопку **Use it** под картинкой-примером, и изображение загрузится в рабочую область как обычный файл |
                | ✅ **Текущий пример** | Кнопка **Using** показывает, какой пример выбран сейчас |
                | ⚠️ **Свой файл** | Если загружен собственный снимок через загрузчик, кнопки примеров недоступны — сначала очистите загрузчик (для этого нажмите :material/close: на загрузчике)|

                ### **Работа с оверлеем** 
                | Действие | Описание |
                |----------|----------|
                | 🔍 **Масштабирование** | Наведите курсор на рабочую область (внутри серой рамки) и используйте колесо мыши |
                | ↩️ **Сброс масштаба** | Выполните двойной клик левой кнопкой мыши по изображению |
                | 🖱️ **Информация об объекте** | Наведите курсор на распознанный объект: он подсветится, появится подсказка с параметрами |
                | 👁️ **Отображение масок** | Щелкните правой кнопкой мыши по изображению для переключения видимости всех масок |
            """)
        elif language == "EN":
            st.markdown("""
                ### ** Example images**
                | Action | Description |
                |--------|-------------|
                | ▶️ **Run processing** | Click **Use it** under an example image — it will be loaded into the workspace just like a regular upload |
                | ✅ **Current example** | The **Using** button indicates which example is currently selected |
                | ⚠️ **Own file** | If a custom image is uploaded via the uploader, the example buttons are disabled — clear the uploader first (click on :material/close: in uploader)|

                ### **Using the overlay**
                | Action | Description |
                |--------|-------------|
                | 🔍 **Zoom** | Place the cursor inside the workspace (within the gray border) and use the mouse wheel |
                | ↩️ **Reset zoom** | Double-click the left mouse button on the image |
                | 🖱️ **Object information** | Hover over a detected object: it will be highlighted and a tooltip with its parameters will appear |
                | 👁️ **Show or hide masks** | Right-click on the image to toggle the visibility of all object masks |
            """)
            
    def help(language="RU"):
        if language == "RU":
            st.markdown("""
                📖 **Руководство** — подробная инструкция по работе с Biofilm Analyzer доступна [здесь](https://disk.yandex.ru/i/67FqW7pGcJ6ELg).

                🖼️ **Примеры изображений** — примеры SEM-снимков доступны [здесь](https://disk.yandex.ru/d/sp1UwEoEBgbyCw).

                ♻️ **Очистка кэша** — если сайт работает некорректно, попробуйте очистить кэш с помощью кнопки ниже.

                ✉️ **Обратная связь** — pawlova12@yandex.ru
            """)
        else: 
            st.markdown("""
                📖 **User manual** — a detailed user guide is available [here](https://disk.yandex.ru/i/67FqW7pGcJ6ELg).

                🖼️ **Image examples** — examples of SEM images are available [here](https://disk.yandex.ru/d/sp1UwEoEBgbyCw).

                ♻️ **Clear cache** — if the website is not working correctly, try clearing the cache using the button below.

                ✉️ **Contact** — pawlova12@yandex.ru
            """)

    def noStatistisResults():
        st.info("No results for statistics calculation.")
        
    def needUploadImageToAnnotate():
        st.info("Upload an image first to import annotations.")

class Error:

    def cantMatchAnnotation():
        st.error(f"Can't render annotation correctly. Check if image size is match with annotation size.")