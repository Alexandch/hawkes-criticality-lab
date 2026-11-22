"""Append a verified R12 supplement to the immutable R1-R11 PDF edition.

Run with Python containing reportlab and pypdf. The original PDF comes from
the recorded Git revision, or from an explicitly supplied --base-pdf.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,subprocess
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,
    Table,TableStyle,PageBreak,NextPageTemplate,Image)
from pypdf import PdfReader,PdfWriter


BASE_REVISION='c4c75db3af5dcfaca50209c96cdb57baadcf1b24'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base-pdf',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];temporary=root/'tmp/pdfs/r12';temporary.mkdir(parents=True,exist_ok=True)
    base=args.base_pdf or temporary/'base_r1_r11.pdf'
    if args.base_pdf is None:base.write_bytes(subprocess.check_output(['git','show',BASE_REVISION+':output/pdf/hawkes_criticality_report.pdf'],cwd=root))
    original=PdfReader(base);offset=len(original.pages)
    out=root/'results/expanded_district_detector'
    with (out/'headline_all.csv').open(encoding='utf-8',newline='') as f:headline=list(csv.DictReader(f))
    with (out/'headline.csv').open(encoding='utf-8',newline='') as f:all_slices=list(csv.DictReader(f))
    with (out/'data_summary.csv').open(encoding='utf-8',newline='') as f:summary=list(csv.DictReader(f))
    assessment=json.loads((out/'assessment.json').read_text(encoding='utf-8'))
    fonts=Path('C:/Windows/Fonts')
    for name,file in [('R12Times','times.ttf'),('R12Bold','timesbd.ttf'),('R12Italic','timesi.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('R12Times',normal='R12Times',bold='R12Bold',italic='R12Italic',boldItalic='R12Bold')
    body=ParagraphStyle('Body',fontName='R12Times',fontSize=11,leading=15,spaceAfter=9)
    small=ParagraphStyle('Small',parent=body,fontSize=9,leading=12,spaceAfter=7)
    heading=ParagraphStyle('Heading',fontName='R12Bold',fontSize=18,leading=23,spaceAfter=14)
    sub=ParagraphStyle('Sub',fontName='R12Bold',fontSize=13,leading=17,spaceAfter=10)
    cell=ParagraphStyle('Cell',fontName='R12Times',fontSize=8,leading=10)
    caption=ParagraphStyle('Caption',parent=small,alignment=TA_CENTER)
    def paragraph(text,style=body):return Paragraph(text,style)
    def table(rows,widths):
        content=[[paragraph(str(c),cell) for c in row] for row in rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
            ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c3cdd6')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        return t
    supplement=temporary/'supplement.pdf';doc=BaseDocTemplate(str(supplement),pagesize=A4,title='Дополнение R12 от 22 ноября 2026',author='Hawkes Criticality Lab')
    def footer(canvas,document):
        width,height=canvas._pagesize;canvas.saveState();canvas.setFont('R12Times',8)
        canvas.setFillColor(colors.HexColor('#53606b'));canvas.drawString(18*mm,12*mm,'Hawkes Criticality Lab - дополнение R12, 22.11.2026')
        canvas.drawRightString(width-18*mm,12*mm,str(offset+document.page));canvas.restoreState()
    templates=[]
    for name,size in [('portrait',A4),('landscape',landscape(A4))]:
        w,h=size;templates.append(PageTemplate(id=name,pagesize=size,frames=[Frame(18*mm,23*mm,w-36*mm,h-41*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
    doc.addPageTemplates(templates);story=[]
    story+=[paragraph('Дополнение R12. Расширение данных и проверка районного детектора',heading),
        paragraph('Редакция от 22 ноября 2026 года. Расчёты R12 выполнены на снимках 19 ноября. Основная часть отчёта R1-R11 сохранена как зафиксированная редакция; это дополнение содержит новый опыт и его ограничения.'),
        paragraph('Данные и постановка',sub)]
    total=sum(int(r['common_scope_cases']) for r in summary)
    def count(value):return f'{int(value):,}'.replace(',',' ')
    def city_name(value):return 'New York' if value=='nyc' else value.replace('_',' ').title()
    story.append(paragraph(f'Общая панель увеличена до {count(total)} зарегистрированных случаев по 114 полицейским округам четырёх городов за январь 2019 - сентябрь 2026. Это совокупный объём обучения, настройки, калибровки и проверки. Обучающая часть содержит {count(sum(int(r["train_cases"]) for r in summary))} случай. Проверочные события не объявляются обучающей выборкой.'))
    story.append(table([['Город','Округа','Случаи в общем периоде','Случаи обучения']]+[[city_name(r['city']),r['districts'],count(r['common_scope_cases']),count(r['train_cases'])] for r in summary],[135,55,145,145]))
    story+=[Spacer(1,10),paragraph('Сохраняется масштаб «район × 6 часов». Цель R11: существенное превышение причинного фона, то есть не менее двух дополнительных событий и не менее 50% сверх фона. Отдельно рассматриваются все выбранные категории, имущественные события и нападения/ограбления. Общая группа включает две остальные, поэтому группы не являются независимыми повторениями.'),
        paragraph('Добавлены наблюдения NOAA GHCN-Daily: максимальная и минимальная температура, осадки и снег с флагами отсутствия. Используется погода пятидневной давности; в стресс-сценарии - десятидневной. Фактическая погода будущего интервала не используется. Исторические даты публикации NOAA не восстановлены.',small),
        paragraph('Даты и районы берутся в той точности, в которой они зарегистрированы источником. Предсказание точного времени или участников отдельного преступления не выполняется.',small),
        PageBreak(),paragraph('R12: методы, проверка и результат',heading),
        paragraph('Заранее заданы календарный и адаптивный фон, локальная нейросеть, общая сеть четырёх городов, общая сеть с погодой и два варианта бустинга. Нейросети объединяют три seed. Выбор метода для города и группы выполняется по average precision валидации до вычисления тестовых метрик.'),
        paragraph('Обучающие события - 2019-2023; валидация - первое полугодие 2024; калибровка - последующие прошлые блоки 2024. Для NYPD учитывается квартальная публикация, и последняя калибровка ограничена августом-сентябрём. Проверяются 2025 и январь-сентябрь 2026. Периоды Chicago/SF уже просматривались ранее; районная проверка NYC/Seattle выполняется впервые. Это ретроспективный опыт, а не реально выданные в прошлом прогнозы.',small)]
    names={'seasonal':'Календарь','recent_calendar':'Адаптивный фон','local_neural':'Локальная НС','pooled_neural':'Общая НС','weather_neural':'НС + погода','boosted15':'Бустинг 15','boosted31':'Бустинг 31'}
    rows=[['Город','Период','Выбранный метод','Верные тревоги','Полнота','FPR']]
    def percent(value):return f'{100*float(value):.1f}%' if value and math.isfinite(float(value)) else 'не опр.'
    for r in headline:rows.append([city_name(r['city']),'2025' if r['split']=='test2025' else '2026 I-IX',names[r['method']],percent(r['precision']),percent(r['recall']),percent(r['fpr'])])
    story.append(table(rows,[88,62,116,76,68,70]));story.append(Spacer(1,10))
    passed=assessment['novel_city_all_group_slices_passing'];n=assessment['novel_city_all_group_slices']
    story.append(paragraph(f'Предварительный ориентир: не менее 30% верных тревог, полнота от 10%, FPR до 5%, AP выше выбранного по валидации простого фона и парный интервал улучшения Brier ниже нуля. Среди основных срезов новых городов этому набору соответствует {passed} из {n}. Это инженерный фильтр исследования, а не доказательство пользы внедрения.'))
    total_passed=sum(r['practical_screen_passed'].lower()=='true' for r in all_slices)
    story.append(paragraph(f'Во всех {len(all_slices)} срезах с учётом групп и стресс-сценария фильтр прошли {total_passed}. Есть частичный выигрыш: AP локальной сети Seattle в 2026 равен 0.2258 против 0.1932 адаптивного фона. Такое улучшение ранжирования не обеспечило требуемое качество тревог.',small))
    story.append(paragraph('FPR - доля тревог среди интервалов без целевого всплеска. Доля верных тревог использует другой знаменатель. Интервалы неопределённости получены по 14-суточным блокам со всеми районами вместе; полные районные и месячные таблицы сохранены в репозитории.',small))
    story += [NextPageTemplate('landscape'),PageBreak(),paragraph('R12: рабочие точки выбранных моделей',heading)]
    first=root/'report/assets/expanded_operating.png';story.append(Image(str(first),width=258*mm,height=258*mm/3))
    story.append(paragraph('Рисунок 21. Доли верных тревог, найденных всплесков и ложных тревог выбранных по валидации методов. Красные линии - заранее заданные ориентиры. Данные 2026 предварительные; общий FPR не гарантирует такой же уровень в каждом районе.',caption))
    story += [PageBreak(),paragraph('R12: объединение городов, погода и бустинг',heading)]
    second=root/'report/assets/expanded_ablation.png';story.append(Image(str(second),width=230*mm,height=230*mm*8/13.5))
    story.append(paragraph('Рисунок 22. AP всех семи заранее заданных методов. График показывает описательное сравнение; победитель по тесту не выбирается. Группы, города и периоды имеют разную частоту целевого превышения.',caption))
    story += [NextPageTemplate('portrait'),PageBreak(),paragraph('Доступность источников и границы применения',heading),
        paragraph('Ключевое ограничение NYPD: открытая YTD-выгрузка обновляется поквартально. При проверке 19 ноября последние опубликованные даты регистрации заканчивались 30 сентября. Поэтому дата регистрации не заменяет дату появления записи на портале.'),
        paragraph('Поправки протокола записаны отдельно до просмотра реальных тестовых метрик: публикация NYPD моделируется как конец квартала плюс 45 суток, стресс-вариант - плюс 90 суток. Окно наблюдаемой истории отнесено на 150 суток назад, в стресс-варианте - на 210, чтобы неопубликованные записи не превращались в нулевые количества. Эти допущения не выдаются за восстановленные фактические версии архива. Локальные модели NYC и общие сети переобучены; неизменённые локальные модели других городов сохранены.'),
        paragraph('В Seattle начат журнал фактического first-seen: 2 402 публичные строки ноябрьского окна сохранены 19 ноября. Отсутствие строк NYPD за то же окно не означает отсутствие преступлений. Разовый журнал ещё не является проспективной проверкой качества прогнозов.'),
        paragraph('Для практического применения остаются необходимыми: своевременный поток с проверяемой доступностью; заранее сохранённые будущие прогнозы и накопление зрелых исходов; устойчивое преимущество при согласованном бюджете тревог; проверка дрейфа и районных различий. Внешний полицейский пилот и эффект вмешательства не исследованы.'),
        paragraph('Проверки и источники',sub),
        paragraph('Пройдены 110 программных тестов. Дополнительно воспроизведены сохранённые прогнозы нейросетей и бустинга, проверены хеши исходников, панелей и моделей, восстановлены погодные значения из NOAA и повторно рассчитаны все таблицы R12.',small),
        paragraph('Источники: NYC Open Data, NYPD Complaint Data Historic (qgea-i56i) и Current YTD (5uac-w243); Seattle SPD Crime Data (tazs-3rd5); Chicago Crimes (ijzp-q8t2); San Francisco Incident Reports (wg3w-h783); NOAA GHCN-Daily, описание формата: https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt.',small),
        paragraph('Протокол, замороженный план, показатели и поправка до теста доступны в репозитории: https://github.com/Alexandch/hawkes-criticality-lab. Основной каталог нового опыта: results/expanded_district_detector. Исходы реальных данных не используются как разметка истинной критичности процесса Хоукса.',small)]
    doc.build(story);extra=PdfReader(supplement)
    writer=PdfWriter();writer.append(original);writer.append(extra,import_outline=False);writer.add_outline_item('Дополнение R12 - расширение данных, 22.11.2026',offset)
    writer.add_metadata({'/Title':'Исследование критических режимов - отчёт R1-R12','/Author':'Hawkes Criticality Lab'})
    target=root/'output/pdf/hawkes_criticality_report.pdf'
    with target.open('wb') as handle:writer.write(handle)
    record=dict(base_revision=BASE_REVISION,base_sha256=digest(base),base_pages=offset,supplement_pages=len(extra.pages),
        total_pages=offset+len(extra.pages),output_sha256=digest(target),builder_sha256=digest(Path(__file__)),
        source_sha256={str(p.relative_to(root)).replace('\\','/'):digest(p) for p in [out/'headline_all.csv',out/'headline.csv',out/'data_summary.csv',out/'assessment.json',first,second]})
    (root/'output/pdf/r12_build.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
