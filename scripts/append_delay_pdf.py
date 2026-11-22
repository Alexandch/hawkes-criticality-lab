"""Append R14 without rewriting the preserved 57-page R1-R13 edition."""
from __future__ import annotations
import csv,hashlib,json,subprocess
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Table,TableStyle,PageBreak,NextPageTemplate,Image,Spacer
from pypdf import PdfReader,PdfWriter

BASE_REVISION='89c2cf681313ec067aecf588cc32134d9da2586f'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];temporary=root/'tmp/pdfs/r14';temporary.mkdir(parents=True,exist_ok=True)
    base=temporary/'base_r1_r13.pdf';base.write_bytes(subprocess.check_output(['git','show',BASE_REVISION+':output/pdf/hawkes_criticality_report.pdf'],cwd=root))
    original=PdfReader(base);offset=len(original.pages);assert offset==57
    out=root/'results/delay_forecast_detector'
    with (out/'metrics.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'))
    for name,file in [('R14Times','times.ttf'),('R14Bold','timesbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('R14Times',normal='R14Times',bold='R14Bold',italic='R14Times',boldItalic='R14Bold')
    body=ParagraphStyle('Body',fontName='R14Times',fontSize=11,leading=15,spaceAfter=10)
    small=ParagraphStyle('Small',parent=body,fontSize=9.5,leading=13,spaceAfter=9)
    heading=ParagraphStyle('Heading',fontName='R14Bold',fontSize=18,leading=23,spaceAfter=15)
    cell=ParagraphStyle('Cell',fontName='R14Times',fontSize=9,leading=12)
    def p(text,style=body):return Paragraph(text,style)
    def table(values,widths):
        result=Table([[p(str(c),cell) for c in row] for row in values],colWidths=widths,repeatRows=1)
        result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c3cdd6')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));return result
    supplement=temporary/'supplement.pdf';doc=BaseDocTemplate(str(supplement),pagesize=A4,title='Дополнение R14 от 22 ноября 2026',author='Hawkes Criticality Lab')
    def footer(canvas,document):
        w,h=canvas._pagesize;canvas.saveState();canvas.setFont('R14Times',8);canvas.setFillColor(colors.HexColor('#53606b'))
        canvas.drawString(18*mm,12*mm,'Hawkes Criticality Lab - новый подход R14, 22.11.2026')
        canvas.drawRightString(w-18*mm,12*mm,str(offset+document.page));canvas.restoreState()
    for name,size in [('portrait',A4),('landscape',landscape(A4))]:
        w,h=size;doc.addPageTemplates(PageTemplate(id=name,pagesize=size,frames=[Frame(18*mm,23*mm,w-36*mm,h-41*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
    story=[p('Дополнение R14. Новый подход и отдельный год проверки',heading),
        p('Austin, девять патрульных секторов, интервал 6 часов. Проверяются коррекция задержек регистрации, нейросетевое распределение количества NB2 и настоящие архивные прогнозы температуры GFS с номинальным упреждением 24 часа.'),
        p('Загружены только номера отчётов, даты и время события и регистрации, сектор и категория. Используются Theft, Auto Theft, Burglary, Aggravated Assault и Robbery. Адреса, индивидуальные координаты и персональные характеристики не запрашивались. Сектора AD, BA, CH, DA, ED, FR, GE, HE, ID заданы до проверки.',small),
        p('Доступное историческое число записей сравнивается с ожидаемым уже опубликованным числом. Функция задержки оценена по зрелым когортам до июля 2023 года, с объединением информации между секторами. Это коррекция неполноты зарегистрированных отчётов; число никогда не зарегистрированных преступлений она не восстанавливает.',small),
        p('Три простых метода сравниваются с BCE-сетью и NB2-сетями с погодой и без неё. NB2 оценивает среднее и меняющуюся дисперсию: Var(N) = μ + αμ². Обучены девять сетей, по три seed на вариант. Для всех одна цель превышения скорректированного фона по формуле R11. Новый город и новый фон не допускают прямого сравнения процентов с Seattle.',small)]
    story.append(table([['Назначение','Период событий','Случаев'],
        ['Разработка всего','апрель 2021 - декабрь 2024','136 581'],['Обучение весов','октябрь 2021 - декабрь 2023','83 819'],
        ['Контрольная точка сети','февраль - март 2024','5 878'],['Выбор модели / уровня','июль - сентябрь 2024','8 328'],
        ['Подтверждение без смены выбора','октябрь - ноябрь 2024','5 760'],['Новый финальный тест','весь 2025 год','33 233']],[170,235,88]))
    story += [Spacer(1,12),p('Первая строка включает остальные части разработки и не суммируется с ними. Другие месяцы используются для разогрева, калибровки и созревания меток. Вся панель содержит 169 814 случаев. Веса и правила зафиксированы до загрузки преступлений Austin за 2025 год.',small),
        PageBreak(),p('R14: выбор стратегии и новый контроль',heading),
        p('По июлю-сентябрю 2024 проверены 12 заранее заданных сочетаний: шесть методов и номинальные уровни 2.5/5%. Ни один не совместил FPR до 5% и полноту от 10%. По резервному правилу выбран BCE-ансамбль с погодой: полнота 6.3%, FPR 3.6% на выборе. Подтверждение октября-ноября не меняет выбор; в нём сеть не выдала ни одной тревоги.',small),
        p('Новый 2025 год содержит 13 140 ячеек и 1 497 превышений (11.4%). Ежемесячная калибровка использует только доступные прошлые метки: окно 180 суток заканчивается за 28 суток до обновления. Первая половина калибрует вероятности, вторая - пороги. Веса сетей в тесте не меняются.',small)]
    values=[['Метод','TP / тревог','Верные тревоги','Полнота','FPR']]
    for role,label in [('selected','Выбранная BCE + погода'),('baseline','Фон с коррекцией'),('weather_ablation','NB2 без погоды'),('reporting_ablation','Фон без коррекции')]:
        row=next(r for r in rows if r['scenario']=='primary' and r['group']=='all' and r['role']==role)
        values.append([label,row['true_positives']+' / '+row['alarms'],*[f'{100*float(row[key]):.1f}%' for key in ['precision','recall','fpr']]])
    story += [table(values,[166,84,91,77,75]),Spacer(1,12),
        p('Доля верных тревог = TP / (TP + FP); полнота = TP / число превышений; FPR = FP / число ячеек без превышения. Выбранная и нейросетевая роли совпадают и не образуют независимых повторений. Дополнительный NB2-вариант задан до теста и не заменяет основной по результатам 2025 года.',small),
        p('Для выбранной сети 95% условные интервалы блочного bootstrap: верные тревоги 14.5-17.9%, полнота 12.8-26.6%, FPR 8.6-18.2%. Использованы 27 временных блоков по 14 суток, последний короче, 1 000 повторов. Блок включает все секторы; пространственные ячейки не считаются независимыми.',small),
        p('У NB2 без погоды AP = 0.2001 против 0.1560 у простого фона; Brier = 0.09747 против 0.09932. Интервал парной разности Brier [-0.00248; -0.00116] условно поддерживает выигрыш вероятностных оценок. Но точность тревог 23.4% ниже 30%, а FPR 5.8% выше 5%.',small),
        p('При лагах истории и публикации 14 суток выбранная сеть имеет точность 15.5%, полноту 20.4%, FPR 14.3%. Ни один из 30 зависимых срезов ролей, групп и сценариев не прошёл совместный фильтр. Пороги по тесту не перенастраивались.',small),
        NextPageTemplate('landscape'),PageBreak(),p('R14: рабочие точки на полном новом годе',heading)]
    first=root/'report/assets/delay_forecast_operating.png';second=root/'report/assets/delay_forecast_monthly.png'
    story += [Image(str(first),width=258*mm,height=258*mm*4.4/13),Spacer(1,10),
        p('Рисунок 25. Austin 2025. Полосы - 95% условные интервалы блочного bootstrap; пунктир - заранее заданные ориентиры. Верные тревоги и FPR имеют разные знаменатели. Лучшее ранжирование дополнительной NB2-сети не подтверждает готовность тревог к применению.',small),
        p('Сравнение NB2 без погоды с выбранной BCE меняет одновременно признаки и функцию потерь. По этой таблице нельзя изолированно заключить, что именно погода ухудшает результат.',small),
        PageBreak(),p('R14: нестабильность по месяцам',heading),
        Image(str(second),width=258*mm,height=258*mm*4.3/12),Spacer(1,10),
        p('Рисунок 26. При заранее заданном обновлении калибровки FPR и полнота меняются по месяцам. Годовое усреднение скрывает периоды очень высокой частоты ложных тревог и периоды почти полного отсутствия обнаружений. Калибровочный уровень не является гарантией FPR на новом распределении.',small),
        NextPageTemplate('portrait'),PageBreak(),p('R14: вывод, ограничения и воспроизводимость',heading),
        p('Получен новый количественный результат: распределительная NB2-сеть без погоды лучше ранжирует превышения и оценивает вероятности относительно оптимизированного простого метода на полном отложенном годе нового города. Основная заранее выбранная стратегия и дополнительные варианты не подтвердили совместную полезность тревог.'),
        p('Это результат конкретного опыта. Он не доказывает невозможность любого прогноза и не означает, что все доступные данные использованы. Перенастройка по Austin 2025 потребует другого внешнего контрольного периода; этот год после просмотра больше не является новым тестом.'),
        p('Ограничения доступности. Austin обновляет открытые данные еженедельно; исторические first-seen отдельных строк неизвестны. Приняты 7 суток публикации и стресс 14 суток. Дата регистрации не равна дате появления в портале. Ретроспективные исправления и смещение регистрации сохраняются. Старые границы секторов не реконструированы.',small),
        p('Погода. Open-Meteo Previous Runs содержит прогнозы GFS, а не фактическую будущую погоду. Средняя температура, диапазон и пропуски агрегированы по местному времени с учётом DST; номинальный выпуск должен быть минимум за 12 часов до интервала. Точная доставка потребителю неизвестна. Январь 2024 имеет пропуски и исключён из выбора контрольной точки; 2025 имеет полное покрытие.',small),
        p('Исследование ретроспективное. Проспективно выданных прогнозов и внешнего полицейского пилота не было. Превышение зарегистрированного количества над фоном не является истинной математической критичностью и не прогнозирует отдельное преступление.',small),
        p('Проверка. Пройдены 126 программных тестов. Отдельно воспроизведены исходные случаи и погодные признаки, функция задержки, все 12 кандидатов, вывод девяти сохранённых моделей, ежемесячные политики двух сценариев, два файла прогнозов и все 30 общих, 270 секторных и 360 месячных строк метрик.',small),
        p('Материалы: results/delay_forecast_detector; scripts/verify_delay_forecast.py. Единственный README содержит полный отчёт. Сырые CSV, погодные ответы и большие массивы внешние; опубликованы веса, правила, метрики, контрольные суммы и журнал фиксации плана.',small),
        p('Источники: City of Austin Crime Reports (fdj4-gpfu); Open-Meteo Previous Runs API, GFS Global. Документация: https://open-meteo.com/en/docs/previous-runs-api. Репозиторий: https://github.com/Alexandch/hawkes-criticality-lab.',small)]
    doc.build(story);extra=PdfReader(supplement);assert len(extra.pages)==5
    writer=PdfWriter();writer.append(original);writer.append(extra,import_outline=False)
    writer.add_outline_item('Дополнение R14 - новый подход и Austin 2025',offset)
    writer.add_metadata({'/Title':'Исследование критических режимов - отчёт R1-R14','/Author':'Hawkes Criticality Lab'})
    target=root/'output/pdf/hawkes_criticality_report.pdf'
    with target.open('wb') as handle:writer.write(handle)
    record=dict(base_revision=BASE_REVISION,base_pages=offset,base_sha256=digest(base),supplement_pages=len(extra.pages),total_pages=offset+len(extra.pages),
        output_sha256=digest(target),builder_sha256=digest(Path(__file__)),source_sha256={p.relative_to(root).as_posix():digest(p) for p in
        [out/'metrics.csv',out/'frozen_plan.json',out/'assessment.json',out/'data_summary.csv',out/'verification.json',first,second]})
    (root/'output/pdf/r14_build.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
