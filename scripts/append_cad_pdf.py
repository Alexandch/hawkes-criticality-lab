"""Append an explicitly exploratory R16 to the preserved66-page report."""
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

BASE_REVISION='38171525d53ca5e70cac18d4c1ffecba0b8b64eb'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];temporary=root/'tmp/pdfs/r16';temporary.mkdir(parents=True,exist_ok=True)
    base=temporary/'base_r1_r15.pdf';base.write_bytes(subprocess.check_output(['git','show',BASE_REVISION+':output/pdf/hawkes_criticality_report.pdf'],cwd=root))
    original=PdfReader(base);offset=len(original.pages);assert offset==66;out=root/'results/cad_context_detector'
    with (out/'metrics.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    for name,file in [('R16Times','times.ttf'),('R16Bold','timesbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('R16Times',normal='R16Times',bold='R16Bold',italic='R16Times',boldItalic='R16Bold')
    body=ParagraphStyle('Body',fontName='R16Times',fontSize=11,leading=15,spaceAfter=10)
    small=ParagraphStyle('Small',parent=body,fontSize=9.5,leading=13,spaceAfter=9)
    heading=ParagraphStyle('Heading',fontName='R16Bold',fontSize=18,leading=23,spaceAfter=15)
    cell=ParagraphStyle('Cell',fontName='R16Times',fontSize=9,leading=12)
    def p(text,style=body):return Paragraph(text,style)
    def table(values,widths):
        result=Table([[p(str(c),cell) for c in row] for row in values],colWidths=widths,repeatRows=1)
        result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c3cdd6')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));return result
    supplement=temporary/'supplement.pdf';doc=BaseDocTemplate(str(supplement),pagesize=A4,title='Дополнение R16 от 22 ноября 2026',author='Hawkes Criticality Lab')
    def footer(canvas,document):
        w,h=canvas._pagesize;canvas.saveState();canvas.setFont('R16Times',8);canvas.setFillColor(colors.HexColor('#53606b'))
        canvas.drawString(18*mm,12*mm,'Hawkes Criticality Lab - исследовательский CAD-контекст R16, 22.11.2026')
        canvas.drawRightString(w-18*mm,12*mm,str(offset+document.page));canvas.restoreState()
    for name,size in [('portrait',A4),('landscape',landscape(A4))]:
        w,h=size;doc.addPageTemplates(PageTemplate(id=name,pagesize=size,frames=[Frame(18*mm,23*mm,w-36*mm,h-41*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
    story=[p('Дополнение R16. Информация диспетчерских обращений',heading),
        p('<b>Статус: исследовательское сравнение с повторным использованием исходов.</b> Периоды 2025 и января-августа 2026 уже просмотрены в R14/R15. Фиксация новых моделей до расчёта показателей не делает их новым независимым тестом.'),
        p('Новый источник - Austin APD Computer Aided Dispatch Incidents, 22de-7rzg. В нём есть обращения без письменного отчёта о преступлении. Открытый источник обновляется ежемесячно. Оперативный поток вызовов в этой работе недоступен.',small),
        table([['Учёт источника, январь 2021 - август 2026','Инцидентов'],['Все загруженные диспетчерские обращения','1 641 637'],
            ['Девять секторов с полными временными полями','1 614 776'],['Другие секторы / неполные поля','26 861']],[365,128]),Spacer(1,12),
        p('Получены серверные агрегаты по месяцу обращения, месяцу закрытия, сектору и начальной категории. Идентификаторы, адреса, персональные характеристики, индивидуальные координаты и итоговая классификация не загружались. Инициированные сотрудниками инциденты исключены. CAD-обращения не прибавляются к уникальным преступлениям целевой панели и не являются целиком обучающей выборкой.',small),
        p('Тринадцать новых признаков: средние суточные частоты за последний один и три доступных месяца по четырём каналам (все обращения, disturbance, alarm, traffic), их соотношения и возраст доступного месяца. Учитываются только обращения, закрытые в уже доступном месяце. Приняты публикация после конца месяца закрытия плюс 15 суток и стресс плюс 45 суток. Исторические first-seen и исправления неизвестны.',small),
        p('Парное переобучение: три NB2-сети без CAD и три с CAD, seed 701/709/719. Сохранены график обучения R14, функция потерь и определение превышения. Веса обучены на 83 819 случаях октября 2021 - декабря 2023; контрольная точка выбирается по 5 878 случаям февраля-марта 2024. На обеих сторонах одно месячное правило: окно 180 суток, созревание 28 суток, уровень 5%. Основная модель CAD задана заранее.',small),
        PageBreak(),p('R16: результаты парного сравнения',heading),
        p('Повторно используемый январь-август 2026 содержит прежние 8 748 ячеек и 935 превышений. У новых парных сетей другие seed и веса; их числа не заменяют сохранённые результаты сетей R14/R15.',small)]
    values=[['Метод','TP / тревог','Верные тревоги','Полнота','FPR']]
    for method,label in [('base_nb','NB2 без CAD'),('cad_nb','NB2 с CAD'),('delay_nb','Простой фон')]:
        row=next(r for r in rows if r['split']=='reused2026' and r['scenario']=='primary' and r['group']=='all' and r['method']==method)
        values.append([label,row['true_positives']+' / '+row['alarms'],*[f'{100*float(row[key]):.1f}%' for key in ('precision','recall','fpr')]])
    story += [table(values,[166,84,91,77,75]),Spacer(1,12),
        p('С CAD найдено 189 превышений вместо 144, но ложных тревог стало 1 058 вместо 778. FPR вырос с 10.0% до 13.5%, доля верных тревог снизилась с 15.6% до 15.2%. Рост полноты сопровождался ухудшением рабочей точки.',small),
        p('На срезе 2026 AP снизилась с 0.1532 до 0.1431; Brier вырос с 0.09481 до 0.09700. 95% условный интервал разности Brier (CAD минус без CAD) [0.00124; 0.00326]. На разработке 2025: AP 0.1899 против 0.1961, Brier 0.09803 против 0.09767; интервал разности [0.00015; 0.00058]. В обоих стресс-сценариях разность Brier также положительна.',small),
        p('Ни одна из 36 зависимых строк периодов, сценариев, методов и групп не прошла скрининговый фильтр. Интервалы условны на выбранном опыте, не учитывают весь прошлый поиск подходов и не являются новым независимым подтверждением.',small),
        p('Вывод: месячный CAD-контекст в этой модели добавочной пользы не показал. Его временное разрешение и публикационный лаг ограничивают информацию о ближайших шести часах. Это возможная причина результата, а не установленная единственная причина. Польза оперативных вызовов, ежедневных данных и других архитектур не проверена.',small),
        p('Ограничения: зарегистрированные обращения и преступления, поздние исправления, неизвестная историческая публикация, повторное использование исходов. Готовность к внедрению не подтверждена. Для подтверждающего опыта нужен новый заранее отделённый период или город. Проспективной выдачи прогнозов и внешнего пилота не было.',small),
        NextPageTemplate('landscape'),PageBreak(),p('R16: тревоги с дополнительным CAD-контекстом',heading)]
    first=root/'report/assets/cad_operating.png';second=root/'report/assets/cad_paired.png'
    story += [Image(str(first),width=258*mm,height=258*mm*4.4/13),Spacer(1,12),
        p('Рисунок 29. Исследовательское сравнение на уже просмотренном срезе 2026. Полосы - 95% условные интервалы блочного bootstrap, 18 временных блоков, 1 000 повторов. Дополнительные месячные признаки увеличили число найденных превышений и число ложных тревог. Сам по себе объём нового источника не гарантирует полезного сигнала.',small),
        PageBreak(),p('R16: парные разности вероятностного качества',heading),
        Image(str(second),width=258*mm,height=258*mm*4.4/11),Spacer(1,10),
        p('Рисунок 30. AP: больше - лучше; Brier: меньше - лучше. Сравниваются одинаково обученные парные ансамбли с CAD и без CAD. Условные интервалы разности Brier находятся выше нуля, что соответствует ухудшению в данном опыте. Исходы обоих периодов уже использовались в проекте.',small),
        p('Проверка: 139 программных тестов. Независимо воспроизведены исходные агрегаты, признаки, масштабирование только по обучению, контрольный NLL шести сетей, два сценария и все 36 общих, 324 секторных, 360 месячных строк. Проверка вычислений не является независимой проверкой прогнозного качества.',small),
        p('Материалы: results/cad_context_detector; scripts/verify_cad_context.py. Источник: Austin APD Computer Aided Dispatch Incidents, 22de-7rzg, ежемесячная публикация. Репозиторий: https://github.com/Alexandch/hawkes-criticality-lab. Единственный README содержит полный отчёт; сырые агрегаты и большие массивы внешние.',small)]
    doc.build(story);extra=PdfReader(supplement);assert len(extra.pages)==4
    writer=PdfWriter();writer.append(original);writer.append(extra,import_outline=False);writer.add_outline_item('Дополнение R16 - исследовательский месячный CAD-контекст',offset)
    writer.add_metadata({'/Title':'Исследование критических режимов - отчёт R1-R16','/Author':'Hawkes Criticality Lab'})
    target=root/'output/pdf/hawkes_criticality_report.pdf'
    with target.open('wb') as handle:writer.write(handle)
    record=dict(base_revision=BASE_REVISION,base_pages=offset,base_sha256=digest(base),supplement_pages=len(extra.pages),total_pages=offset+len(extra.pages),
        output_sha256=digest(target),builder_sha256=digest(Path(__file__)),source_sha256={p.relative_to(root).as_posix():digest(p) for p in
        [out/'metrics.csv',out/'paired_summary.csv',out/'frozen_plan.json',out/'assessment.json',out/'verification.json',first,second]})
    (root/'output/pdf/r16_build.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
