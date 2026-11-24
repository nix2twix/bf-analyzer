import streamlit as st

class Message:
    @staticmethod
    def whatsNew(language="RU"):
        if language == "RU":
            st.markdown("""
            ### **Пакетная обработка (ZIP)**
                | Действие | Описание |
                |----------|----------|
                | 📦 **Загрузка архива** | Перетащите ZIP-архив с изображениями в загрузчик (поддерживаются BMP, PNG, JPG, TIFF). Сайт автоматически распознает архив и переключится в режим пакетной обработки |
                | 🚀 **Запуск** | Нажмите **Start batch segmentation**. Прогресс отображается для всего пакета и для каждой картинки отдельно |
                | 📊 **Результаты** | После обработки появится сводка по каждому файлу. Кнопки экспорта позволяют скачать CVAT-бэкап и сводный Excel-отчёт |
                | ⚠️ **Без фильтрации** | В пакетном режиме фильтрация **отключена**: экспортируются все объекты, найденные моделью. Это сделано намеренно, так как настраивать фильтры для десятков картинок вручную неудобно |
                ### **Примеры изображений**
                | Действие | Описание    |
                |----------|-------------|
                | ▶️ **Cегментация** | Нажмите кнопку **Use it** под картинкой-примером, и изображение загрузится в рабочую область как обычный файл |
                | ✅ **Текущий пример** | Кнопка **Using** показывает, какой пример выбран сейчас |
                | ⚠️ **Свой файл** | Если загружен собственный снимок через загрузчик, кнопки примеров недоступны — сначала очистите загрузчик (для этого нажмите :material/close: на загрузчике)|
            """)
        elif language == "EN":
            st.markdown("""
                | Action | Description |
                |--------|-------------|
                | 📦 **Upload archive** | Drag a ZIP archive with images into the uploader (BMP, PNG, JPG, TIFF supported). The app detects the archive and switches to batch mode automatically |
                | 🚀 **Start** | Click **Start batch segmentation**. Progress is shown both for the whole batch and for each image |
                | 📊 **Results** | A summary appears after processing. Export buttons let you download a CVAT backup and a combined Excel report |
                | ⚠️ **No filtering** | Filtering is **disabled** in batch mode: all objects detected by the model are exported. This is intentional, since tuning filters for dozens of images manually is impractical |
                ### ** Example images**
                | Action | Description |
                |--------|-------------|
                | ▶️ **Run processing** | Click **Use it** under an example image — it will be loaded into the workspace just like a regular upload |
                | ✅ **Current example** | The **Using** button indicates which example is currently selected |
                | ⚠️ **Own file** | If a custom image is uploaded via the uploader, the example buttons are disabled — clear the uploader first (click on :material/close: in uploader)|
            """)
    @staticmethod
    def citationAndContacts():
        st.markdown("""
    **Contact:** pawlova12@yandex.ru

    **Citation:**

    > Segmentation of Microscopic Images of Pseudomonas Biofilms Using Deep Learning and Semantic Filtering / V. S. Pavlova, M. Yu. Kurbakov, A. V. Kopylov, O. S. Seredin // Pattern Recognition and Image Analysis. Advances in Mathematical Theory and Applications. – 2026. – Vol. 36, No. 2. – P. 777-792. – DOI [10.1134/S1054661826700586](https://doi.org/10.1134/S1054661826700586).
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
        st.error(f"Can't render annotation correctly. Check names or check if image size is match with annotation size.")