import streamlit as st


class GuideContent:
    """Содержимое вкладок страницы Guide. Разделено по языкам и разделам."""
    # ---------- Quick start ----------
    @staticmethod
    def quickStart(language="EN"):
        if language == "RU":
            st.markdown("""
            ### Быстрый старт
            <table style="width: 100%;">
            <tr>
                <th style="width: 28%;">Шаг</th>
                <th>Действие</th>
            </tr>
            <tr>
                <td>1️⃣ <b>Загрузите изображение</b></td>
                <td>Загрузите СЭМ-изображение в загрузчик справа или выберите пример в разделе <b>Examples</b>. Для пакетной обработки загрузите ZIP-архив с изображениями.</td>
            </tr>
            <tr>
                <td>2️⃣ <b>Выберите модель</b></td>
                <td>Модель Bacillus предназначена для палочковидных микроорганизмов, модель Coccus — для шаровидных.</td>
            </tr>
            <tr>
                <td>3️⃣ <b>Запустите обработку</b></td>
                <td>Для одного изображения нажмите <b>🧪 Start segmentation</b>. Для архива выберите нужные файлы и нажмите <b>🚀 Start batch segmentation</b>.</td>
            </tr>
            <tr>
                <td>4️⃣ <b>Проверьте результаты</b></td>
                <td>Для одного изображения настройте фильтры на вкладке <b>Statistics & Tools</b>. В пакетном режиме фильтры недоступны: результаты и статистика формируются отдельно для каждого изображения.</td>
            </tr>
            <tr>
                <td>5️⃣ <b>Экспортируйте результаты</b></td>
                <td>Нажмите <b>⬇️ Save changes</b>, затем выберите <b>📥 CVAT backup</b> для экспорта масок или <b>💾 Results</b> для получения архива с результатами и статистикой.</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
                ### Quick start
                <table style="width: 100%;">
                <tr>
                    <th style="width: 25%;">Step</th>
                    <th>Action</th>
                </tr>
                <tr>
                    <td>1️⃣ <b>Upload an image</b></td>
                    <td>Upload a SEM image using the uploader on the right, or choose an example from the <b>Examples</b> section. For batch processing, upload a ZIP archive with images.</td>
                </tr>
                <tr>
                    <td>2️⃣ <b>Choose a model</b></td>
                    <td><b>Bacillus</b> is intended for rod-shaped microorganisms; <b>Coccus</b> is intended for spherical ones.</td>
                </tr>
                <tr>
                    <td>3️⃣ <b>Run processing</b></td>
                    <td>For a single image, click <b>🧪 Start segmentation</b>. For an archive, select the files to process and click <b>🚀 Start batch segmentation</b>.</td>
                </tr>
                <tr>
                    <td>4️⃣ <b>Review the results</b></td>
                    <td>For a single image, adjust filters in <b>Statistics & Tools</b>. Filters are unavailable in batch mode; results and statistics are calculated separately for each image.</td>
                </tr>
                <tr>
                    <td>5️⃣ <b>Export results</b></td>
                    <td>Click <b>⬇️ Save changes</b>, then choose <b>📥 CVAT backup</b> to export masks or <b>💾 Results</b> to download an archive with results and statistics.</td>
                </tr>
                </table>
                """, unsafe_allow_html=True)

    # ---------- Batch ----------
    @staticmethod
    def batch(language="RU"):
        if language == "RU":
            st.markdown("""
            ### Пакетная обработка
            <table style="width: 100%;">
            <tr>
            <th style="width: 20%;">Действие</th>
            <th>Описание</th>
            </tr>
            <tr>
            <td>📦 <b>Загрузка архива</b></td>
            <td>Загрузите ZIP-архив с СЭМ-изображениями в загрузчик на вкладке <b>Segmentation</b>. Приложение автоматически переключится в пакетный режим.</td>
            </tr>
            <tr>
            <td>☑️ <b>Выбор изображений</b></td>
            <td>В рабочей области появится предпросмотр изображений: по четыре миниатюры на странице с постраничной навигацией. Отметьте чекбоксами изображения, которые нужно обработать. Выбор сохраняется при переходе между страницами. Кнопки <b>Select all</b> и <b>Deselect all</b> выбирают или снимают выбор со всех изображений архива.</td>
            </tr>
            <tr>
            <td>🚀 <b>Запуск обработки</b></td>
            <td>Выберите режим определения масштаба и нажмите <b>Start batch segmentation</b>. Отображается общий прогресс обработки и прогресс обработки фрагментов текущего изображения.</td>
            </tr>
            <tr>
            <td>📊 <b>Результаты</b></td>
            <td>После обработки в рабочей области появится сводка по изображениям. Для каждого изображения отображаются размеры, масштаб и статистика по классам. Изображения с ошибками обработки выносятся отдельно.</td>
            </tr>
            <tr>
            <td>⚠️ <b>Фильтрация</b></td>
            <td>Фильтры площади и формы в пакетном режиме недоступны. Из результатов автоматически исключаются шумовые объекты (площадью менее <b>200 px</b>). Остальные обнаруженные объекты включаются в статистику и экспорт.</td>
            </tr>
            </table>

            ### Определение масштаба

            При пакетной обработке доступны три режима:
            <table style="width: 100%;">
            <tr>
            <th style="width: 20%;">Режим</th>
            <th>Описание</th>
            </tr>
            <tr>
            <td><b>First image only</b></td>
            <td>Масштаб определяется по первому изображению и используется для остальных изображений пакета. Подходит, если все снимки имеют одинаковый масштаб.</td>
            </tr>
            <tr>
            <td><b>Each image</b></td>
            <td>Масштаб определяется отдельно для каждого изображения. Этот режим может увеличить время обработки.</td>
            </tr>
            <tr>
            <td><b>Skip detection</b></td>
            <td>Автоматическое определение масштаба отключено. Физический масштаб считается неизвестным, площади в физических единицах не рассчитываются.</td>
            </tr>
            </table>

            При загрузке CVAT-бэкапа автоматическая сегментация не запускается: используются уже сохранённые маски. Для каждого изображения выполняется попытка определить масштаб отдельно. Если масштаб не найден, в сводке он отмечается как неизвестный, а площади в µm² не рассчитываются.

            ### Экспорт результатов

            - **📥 CVAT backup** — архив с результатами разметки, который можно загрузить в CVAT.
            - **💾 Results** — ZIP-архив со сводным Excel-отчётом. Каждая строка листа `Batch summary` соответствует одному изображению и содержит его размеры, масштаб и статистику по классам. Если при обработке были ошибки, в отчёт также добавляется лист `Errors`.

            ### Импорт разметки из CVAT

            Чтобы использовать разметку, доработанную в CVAT, загрузите CVAT backup на вкладке **Statistics & Tools** в разделе **📂 Import CVAT-annotations**. Приложение загрузит изображения и готовые маски в пакетный режим, пересчитает статистику и попытается определить масштаб для каждого изображения. Повторная сегментация не требуется.
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            ### Batch processing
            <table style="width: 100%;">
            <tr>
            <th style="width: 20%;">Action</th>
            <th>Description</th>
            </tr>
            <tr>
            <td>📦 <b>Upload archive</b></td>
            <td>Upload a ZIP archive containing SEM images using the uploader in the <b>Segmentation</b> tab. The app automatically switches to batch mode.</td>
            </tr>
            <tr>
            <td>☑️ <b>Select images</b></td>
            <td>The workspace displays image previews: four thumbnails per page with pagination. Use the checkboxes to select images for processing. Selection is preserved when switching pages. <b>Select all</b> and <b>Deselect all</b> apply to the entire archive.</td>
            </tr>
            <tr>
            <td>🚀 <b>Start processing</b></td>
            <td>Choose a scale-detection mode and click <b>Start batch segmentation</b>. Progress is displayed for the whole batch and for the patches of the current image.</td>
            </tr>
            <tr>
            <td>📊 <b>Results</b></td>
            <td>A summary appears in the workspace after processing. For each image it shows dimensions, scale, and per-class statistics. Images that failed are listed separately.</td>
            </tr>
            <tr>
            <td>⚠️ <b>Filtering</b></td>
            <td>Area and shape filters are unavailable in batch mode. Objects with an area below <b>200 px</b> are automatically excluded. The remaining detected objects are included in statistics and exports.</td>
            </tr>
            </table>

            ### Scale detection

            Three modes are available for batch processing:
            <table style="width: 100%;">
            <tr>
            <th style="width: 20%;">Mode</th>
            <th>Description</th>
            </tr>
            <tr>
            <td><b>First image only</b></td>
            <td>Detects the scale in the first image and reuses it for the remaining images. Use this mode when all images have the same scale.</td>
            </tr>
            <tr>
            <td><b>Each image</b></td>
            <td>Detects the scale independently for every image. This mode may increase processing time.</td>
            </tr>
            <tr>
            <td><b>Skip detection</b></td>
            <td>Disables automatic scale detection. The physical scale is treated as unknown, so areas in physical units are not calculated.</td>
            </tr>
            </table>

            When a CVAT backup is imported, segmentation is not run again: the masks stored in the backup are reused. The app attempts to detect the scale separately for each image. If the scale cannot be detected, it is marked as unknown and areas in µm² are not calculated.

            ### Exporting results

            - **📥 CVAT backup** — an archive containing the annotation results, ready to be imported into CVAT.
            - **💾 Results** — a ZIP archive with a summary Excel report. Each row in the `Batch summary` sheet represents one image and contains its dimensions, scale, and per-class statistics. If any images failed during processing, an additional `Errors` sheet is included.

            ### Importing annotations from CVAT

            To use annotations refined in CVAT, upload the CVAT backup in the **Statistics & Tools** tab under **📂 Import CVAT-annotations**. The app loads the images and existing masks into batch mode, recalculates statistics, and attempts to detect the scale for each image. Segmentation is not run again.
            """, unsafe_allow_html=True)
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
            <table style="width: 100%;">
            <tr>
            <th style="width: 20%;">Действие</th>
            <th>Описание</th>
            </tr>
            <tr>
            <td>🔍 <b>Масштабирование</b></td>
            <td>С помощью колесика мыши</td>
            </tr>
            <tr>
            <td>↩️ <b>Сброс масштаба</b></td>
            <td>Двойной клик левой кнопкой мыши по изображению</td>
            </tr>
            <tr>
            <td>🖱️ <b>Информация об объекте</b></td>
            <td>Наведите курсор на распознанный объект: он подсветится, появится подсказка с параметрами</td>
            </tr>
            <tr>
            <td>👁️ <b>Отображение масок</b></td>
            <td>Правый клик по изображению осуществляет переключение видимости всех масок</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            ### Using the overlay
            <table style="width: 100%;">
            <tr>
            <th>Action</th>
            <th>Description</th>
            </tr>
            <tr>
            <td>🔍 <b>Zoom</b></td>
            <td>Cursor inside the workspace (within the gray border) + mouse wheel</td>
            </tr>
            <tr>
            <td>↩️ <b>Reset zoom</b></td>
            <td>Double-click the left mouse button on the image</td>
            </tr>
            <tr>
            <td>🖱️ <b>Object information</b></td>
            <td>Hover over a detected object: it highlights and a tooltip with its parameters appears</td>
            </tr>
            <tr>
            <td>👁️ <b>Show or hide masks</b></td>
            <td>Right-click the image to toggle visibility of all object masks</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)

    # ---------- Examples ----------
    @staticmethod
    def examples(language="RU"):
        if language == "RU":
            st.markdown("""
            ### Примеры изображений
            <table style="width: 100%;">
            <tr>
            <th style="width: 20%;">Действие</th>
            <th>Описание</th>
            </tr>
            <tr>
            <td>▶️ <b>Сегментация</b></td>
            <td>Нажмите <b>Use it</b> под картинкой-примером, и изображение загрузится как обычный файл</td>
            </tr>
            <tr>
            <td>✅ <b>Текущий пример</b></td>
            <td>Кнопка <b>Using</b> показывает, какой пример выбран сейчас</td>
            </tr>
            <tr>
            <td>⚠️ <b>Свой файл</b></td>
            <td>Если загружен собственный снимок, кнопки выбора примеров недоступны. Для их активации очистите загрузчик (нажмите в нём :material/close:)</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            ### Example images
            <table style="width: 100%;">
            <tr>
            <th>Action</th>
            <th>Description</th>
            </tr>
            <tr>
            <td>▶️ <b>Run processing</b></td>
            <td>Click <b>Use it</b> under an example image — it will be loaded like a regular upload</td>
            </tr>
            <tr>
            <td>✅ <b>Current example</b></td>
            <td>The <b>Using</b> button indicates which example is selected</td>
            </tr>
            <tr>
            <td>⚠️ <b>Own file</b></td>
            <td>If a custom image is uploaded, example buttons are disabled — clear the uploader (:material/close:)</td>
            </tr>
            </table>
            """, unsafe_allow_html=True)

    # ---------- FAQ ----------
    @staticmethod
    def faq(language="RU"):
        if language == "RU":
            st.markdown("""
            ### Часто задаваемые вопросы

            1. **Где взять больше примеров СЭМ-изображений, с которыми работают модели, представленные на сайте?**
            - Набор данных доступен по ссылке 🖼️ [Примеры СЭМ-снимков](https://disk.yandex.ru/d/sp1UwEoEBgbyCw)

            2. **Почему маски выглядят «рваными» по краям?**
            - Такое случается при выключенной постобработке. Попробуйте включить опцию **Morphological postprocess** во вкладке **Statistics & Tools** → **📊 Tools**. Маски станут более сглаженными, однако время обработки может увеличиться.

            3. **Почему в пакетном режиме нет фильтров?**
            - В пакетном режиме фильтры площади и формы недоступны. Из результатов автоматически исключаются объекты площадью менее **200 px**, остальные обнаруженные объекты включаются в статистику и экспорт. При необходимости скорректировать разметку можно в CVAT.

            4. **Что делать, если масштаб не определился?**
            - Для одиночного изображения проверьте значение масштаба под изображением. В пакетном режиме выберите подходящий режим определения масштаба. Если автоматическое определение не сработало, масштаб будет отмечен как неизвестный, а площади в µm² не будут рассчитаны.

            5. **Куда обратиться за помощью?**
            - Воспользуйтесь документацией или свяжитесь с ✉️ pawlova12@yandex.ru

            6. **Сайт работает некорректно. Что попробовать?**
            - Нажмите **♻ Clear cache** ниже. Это очистит кэш моделей и изображений, не затрагивая ваши данные.
            """)
        else:
            st.markdown("""
            ### FAQ

            1. **Where can I find more SEM images compatible with the models available on the website?**
            - The dataset is available here: 🖼️ [SEM image examples](https://disk.yandex.ru/d/sp1UwEoEBgbyCw)

            2. **Why do masks look ragged at the edges?**
            - This may happen when postprocessing is disabled. Enable **Morphological postprocess** in **Statistics & Tools** → **📊 Tools**. It smooths the masks, but may increase processing time.

            3. **Why are filters unavailable in batch mode?**
            - Area and shape filters are disabled in batch mode. Objects smaller than **200 px** are automatically excluded; all remaining detected objects are included in statistics and exports. You can refine the annotations in CVAT if needed.

            4. **What should I do if the scale is missing or incorrect?**
            - For a single image, check the scale value below the image and enter it manually if necessary. In batch mode, choose the appropriate scale-detection mode. If automatic detection fails, the scale is marked as unknown and areas in µm² are not calculated.

            5. **The site is not working correctly. What can I try?**
            - Click **♻ Clear cache** below. This clears model and image caches without affecting your data.

            6. **Where can I get help?**
            - Contact us at ✉️ pawlova12@yandex.ru
            """)
