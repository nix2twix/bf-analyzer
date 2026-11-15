import streamlit as st


class GuideContent:
    """Содержимое вкладок страницы Guide. Разделено по языкам и разделам."""

    # ---------- Quick start ----------
    @staticmethod
    def quickStart(language="RU"):
        if language == "RU":
            st.markdown("""
            ### Быстрый старт

            | Шаг | Действие |
            |-----|----------|
            | 1️⃣ **Загрузите снимок** | Перетащите SEM-изображение (BMP, PNG, JPG, TIFF) в загрузчик справа или выберите пример ниже |
            | 2️⃣ **Выберите модель** | Bacillus — для палочковидных, Coccus — для шаровидных микроорганизмов |
            | 3️⃣ **Запустите сегментацию** | Нажмите **🧪 Start segmentation**. Дождитесь завершения — на изображении появятся маски объектов |
            | 4️⃣ **Настройте фильтры** | Вкладка **Statistics & Tools** содержит настройки фильтрации по площади и форме организмов. Статистика обновляется автоматически |
            | 5️⃣ **Экспортируйте** | Нажмите **⬇️ Save changes** и выберите **📥 CVAT backup** (маски в CVAT) или **💾 Results** (ZIP с Excel и PNG) |

            💡 Масштаб определяется автоматически по масштабной линейке на снимке. Если не удалось определить масштаб, значения площадей рассчитываются в пикселях.
            """)
            st.space()
        else:
            st.markdown("""
            ### Quick start

            | Step | Action |
            |------|--------|
            | 1️⃣ **Upload an image** | Drop a SEM image (BMP, PNG, JPG, TIFF) into the uploader on the right, or pick an example below |
            | 2️⃣ **Choose a model** | Bacillus — for rod-shaped, Coccus — for spherical microorganisms |
            | 3️⃣ **Run segmentation** | Click **🧪 Start segmentation**. Once done, object masks will appear on the image |
            | 4️⃣ **Tune filters** | Go to **Statistics & Tools**, where is area and eccentricity sliders. Statistics update automatically |
            | 5️⃣ **Export** | Click **⬇️ Save changes**, then **📥 CVAT backup** (masks for CVAT) or **💾 Results** (ZIP with Excel and PNG) |

            💡 Scale is auto-detected from the scale bar on the image. If detection fails, area values will appear in pixel. 
            """)

    # ---------- Batch ----------
    @staticmethod
    def batch(language="RU"):
        if language == "RU":
            st.markdown("""
            ### Пакетная обработка 

            | Действие | Описание |
            |----------|----------|
            | 📦 **Загрузка архива** | Перетащите ZIP с изображениями в загрузчик в правой части рабочей области. Сайт автоматически распознаёт архив и переключается в пакетный (batch) режим |
            | 🚀 **Запуск** | Нажмите **🚀 Start batch segmentation** для запуска пакетной обработки. Отображается прогресс для всего пакета, и для каждой картинки отдельно |
            | 📊 **Результаты** | После обработки появляется сводка. Кнопки экспорта возвращают CVAT-бэкап и сводный Excel-отчёт |
            | ⚠️ **Без фильтрации** | Для оптимизации в batch-режиме фильтрация **отключена**: экспортируются все объекты, найденные моделью. |
            | ⚠️ **Масштаб** | В batch-режиме абсолютные величины масштаба не определяется (принято 1 px = 1 μm). Площади в отчёте указаны в пикселях² |

            ### Работа с результатом

            - **📥 CVAT backup** — содержит N картинок в `data/`, импортируется в CVAT как задача с N кадрами.
            - **💾 Results** — архив ZIP со таблицей Excel, где для каждого изображения указаны классы и % площади
            """)
        else:
            st.markdown("""
            ### Batch processing (ZIP)

            | Action | Description |
            |--------|-------------|
            | 📦 **Upload archive** | Drop a ZIP with images into the same uploader. The app detects the archive and switches to batch mode automatically |
            | 🚀 **Start** | Click **Start batch segmentation**. Progress is shown for the whole batch and for each image |
            | 📊 **Results** | A summary appears after processing. Export buttons produce a CVAT backup and a combined Excel report |
            | ⚠️ **No filtering** | Filtering is **disabled** in batch mode: all objects detected by the model are exported. Intentional — tuning filters for dozens of images manually is impractical |
            | ⚠️ **Scale** | Scale is not detected in batch mode (assumed 1 px = 1 μm). Areas in the report are in px² |

            ### What is in the exported archives

            - **📥 CVAT backup** — N images in `data/`, `manifest.jsonl`, `task.json`, `annotations.json`. Imports into CVAT as a backup task with N frames.
            - **💾 Results** — ZIP with a combined Excel file: sheet **Batch summary** (rows = images, columns = classes and % area), sheet **Filters** (note that filtering is disabled), sheet **Errors** if any files failed.
            """)

    # ---------- Overlay ----------
    @staticmethod
    def overlay(language="RU"):
        if language == "RU":
            st.markdown("""
            ### Работа с оверлеем
            Оверлей - это интерактивное окно, которое появляется после загрузки СЭМ-изображения. Он находится в левой части рабочей области. 
            Взаимодействие с оверлеем происходит посредством мыши. При этом курсор должен находиться в области оверлея (ограничена серой рамкой)
            """)
            st.space()
            st.markdown("""
            | Действие | Описание |
            |----------|----------|
            | 🔍 **Масштабирование** | С помощью колесика мыши |
            | ↩️ **Сброс масштаба** | Двойной клик левой кнопкой мыши по изображению |
            | 🖱️ **Информация об объекте** | Наведите курсор на распознанный объект: он подсветится, появится подсказка с параметрами |
            | 👁️ **Отображение масок** | Правый клик по изображению осуществляет переключение видимости всех масок |
            """)
        else:
            st.markdown("""
            ### Using the overlay

            | Action | Description |
            |--------|-------------|
            | 🔍 **Zoom** | Cursor inside the workspace (within the gray border) + mouse wheel |
            | ↩️ **Reset zoom** | Double-click the left mouse button on the image |
            | 🖱️ **Object information** | Hover over a detected object: it highlights and a tooltip with its parameters appears |
            | 👁️ **Show or hide masks** | Right-click the image to toggle visibility of all object masks |
            """)

    # ---------- Examples ----------
    @staticmethod
    def examples(language="RU"):
        if language == "RU":
            st.markdown("""
            ### Примеры изображений
            | Действие | Описание |
            |----------|----------|
            | ▶️ **Сегментация** | Нажмите **Use it** под картинкой-примером, и изображение загрузится как обычный файл |
            | ✅ **Текущий пример** | Кнопка **Using** показывает, какой пример выбран сейчас |
            | ⚠️ **Свой файл** | Если загружен собственный снимок, кнопки выбора примеров недоступны. Для их активации очистите загрузчик (нажмите в нём :material/close:) |
            """)
        else:
            st.markdown("""
            ### Example images

            | Action | Description |
            |--------|-------------|
            | ▶️ **Run processing** | Click **Use it** under an example image — it will be loaded like a regular upload |
            | ✅ **Current example** | The **Using** button indicates which example is selected |
            | ⚠️ **Own file** | If a custom image is uploaded, example buttons are disabled — clear the uploader (:material/close:) |
            """)

    # ---------- FAQ ----------
    @staticmethod
    def faq(language="RU"):
        if language == "RU":
            st.markdown("""
            ### FAQ

            1. **Где взять больше примеров СЭМ-изображений, с которыми работают модели, представленные на сайте?**
            - Набор данных доступен по ссылке 🖼️ [Примеры SEM-снимков](https://disk.yandex.ru/d/sp1UwEoEBgbyCw)

            2. **Почему маски выглядят «рваными» по краям?**
            - Такое случается при выключенной постобработке. Попробуйте включить опцию **Morphological postprocess** во вкладке **Statistics & Tools**. Тогда маски станут более сглаженными. Обратите внимание, что время обработки увеличится в зависимости от количества найденных объектов.   

            3. **Почему в batch-режиме нет фильтров?**
            - Настраивать ползунки для десятков картинок неудобно и приводит к разным параметрам на разных снимках. В batch экспортируются все объекты, найденные моделью — так отчёт воспроизводим.

            4. **Куда обратиться за помощью?**
            - Воспользуйтесь 📖 [Руководство пользователя](https://disk.yandex.ru/i/67FqW7pGcJ6ELg) или свяжитесь с ✉️ pawlova12@yandex.ru

            5. **Сайт работает некорректно. Что попробовать?**
            - Нажмите **♻ Clear cache** ниже. Это очистит кэш моделей и изображений, не затрагивая ваши данные.

            """)
        else:
            st.markdown("""
            ### FAQ

            **Why do masks look ragged at the edges?**
            Enable **Morphological postprocess** in **Statistics & Tools** → **📊 Tools**. It smooths masks and fills holes.

            **Scale was detected incorrectly. What should I do?**
            Click **🔎 Scale: …** under the image and enter the scale-bar length in micrometers manually. All areas and perimeters will be recalculated.

            **Why are filters unavailable in batch mode?**
            Tuning sliders for dozens of images is impractical and would produce inconsistent parameters across images. Batch exports all objects detected by the model, so the report is reproducible.

            **The site misbehaves. What can I try?**
            Click **♻ Clear cache** below. It clears model and image caches, without touching your data.

            **Where can I get help?**
            📖 [User manual](https://disk.yandex.ru/i/67FqW7pGcJ6ELg) · 🖼️ [SEM image examples](https://disk.yandex.ru/d/sp1UwEoEBgbyCw) · ✉️ pawlova12@yandex.ru
            """)