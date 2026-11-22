"""Append the final R13 attempt to the preserved 53-page R1-R12 edition."""
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


BASE_REVISION='c2bb333ce5a6c99dc7906b5376f5b56c743c11ca'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];temporary=root/'tmp/pdfs/r13';temporary.mkdir(parents=True,exist_ok=True)
    base=temporary/'base_r1_r12.pdf';base.write_bytes(subprocess.check_output(['git','show',BASE_REVISION+':output/pdf/hawkes_criticality_report.pdf'],cwd=root))
    original=PdfReader(base);offset=len(original.pages);assert offset==53
    out=root/'results/adaptive_seattle_detector'
    with (out/'headline.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    frozen=json.loads((out/'frozen_plan.json').read_text(encoding='utf-8'));selection=frozen['roles']['selected']
    for name,file in [('R13Times','times.ttf'),('R13Bold','timesbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('R13Times',normal='R13Times',bold='R13Bold',italic='R13Times',boldItalic='R13Bold')
    body=ParagraphStyle('Body',fontName='R13Times',fontSize=11,leading=15,spaceAfter=9)
    small=ParagraphStyle('Small',parent=body,fontSize=9,leading=12,spaceAfter=8)
    heading=ParagraphStyle('Heading',fontName='R13Bold',fontSize=18,leading=23,spaceAfter=14)
    cell=ParagraphStyle('Cell',fontName='R13Times',fontSize=8,leading=10)
    def p(text,style=body):return Paragraph(text,style)
    supplement=temporary/'supplement.pdf';doc=BaseDocTemplate(str(supplement),pagesize=A4,title='Дополнение R13 от 22 ноября 2026',author='Hawkes Criticality Lab')
    def footer(canvas,document):
        w,h=canvas._pagesize;canvas.saveState();canvas.setFont('R13Times',8);canvas.setFillColor(colors.HexColor('#53606b'))
        canvas.drawString(18*mm,12*mm,'Hawkes Criticality Lab - заключительная попытка R13, 22.11.2026')
        canvas.drawRightString(w-18*mm,12*mm,str(offset+document.page));canvas.restoreState()
    for name,size in [('portrait',A4),('landscape',landscape(A4))]:
        w,h=size;doc.addPageTemplates(PageTemplate(id=name,pagesize=size,frames=[Frame(18*mm,23*mm,w-36*mm,h-41*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
    story=[p('Дополнение R13. Последняя попытка улучшить районные тревоги',heading),
        p('Seattle, район × 6 часов. Веса нейросети R12 сохранены. Проверяется ежемесячное обновление калибровки вероятностей и районных порогов по доступным прошлым данным.'),
        p('Заранее заданы 27 вариантов: три score-модели, окна 90/180/365 суток и уровни 1/2.5/5%. Окно заканчивается за 28 суток до обновления; его первая половина калибрует вероятности, вторая - пороги. Метки учитывают дату регистрации плюс условные 3 суток публикации, в стресс-сценарии - 7 суток.',small),
        p('Совместный выбор модели и порога выполнен по уже просмотренным 2025 и январю-сентябрю 2026. Эти периоды являются разработкой. Ни один вариант не сочетал FPR до 5% и полноту от 10% в обоих периодах. По заранее заданному резервному правилу выбрана нейросеть с 90-суточным окном и калибровочным уровнем 2.5%.',small),
        p('Новый контроль - октябрь 2026, загруженный после фиксации правил: 3 708 строк, 3 564 новых случая. Историческая панель R12 не изменена. В полном октябре 620 районных ячеек и 77 целевых превышений; в первых 21 сутках - 420 и 44 соответственно.',small)]
    roles=[('static_neural','НС R12'),('rolling_neural','НС, обновление'),('static_adaptive','Фон R12'),('rolling_baseline','Фон, обновление')]
    table=[['Период','Метод','Найдено / всего','Тревог','Верные тревоги','Полнота','FPR']]
    def percent(value):return f'{100*float(value):.1f}%' if value else 'не опр.'
    for split,label in [('new_october_all','Октябрь'),('new_october_first21','1-21 окт.')]:
        for role,name in roles:
            row=next(r for r in rows if r['split']==split and r['role']==role)
            table.append([label,name,row['true_positives']+' / '+row['positives'],row['alarms'],percent(row['precision']),percent(row['recall']),percent(row['fpr'])])
    rendered=Table([[p(c,cell) for c in row] for row in table],colWidths=[52,105,72,48,75,65,63],repeatRows=1)
    rendered.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c3cdd6')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story+=[rendered,Spacer(1,10),p('Доля верных тревог равна TP / (TP + FP), полнота - TP / число превышений. FPR использует только ячейки без превышения. Выбранная основная стратегия совпадает с обновляемой нейросетью; это одна стратегия, а не независимый дополнительный опыт.',small),NextPageTemplate('landscape'),PageBreak(),p('R13: новый контроль и рабочая точка',heading)]
    first=root/'report/assets/adaptive_october.png';story.append(Image(str(first),width=258*mm,height=258*mm/3))
    story.append(p('Рисунок 23. Октябрьский контроль Seattle. Калибровка и пороги зафиксированы до загрузки новых исходов. Уменьшение ложных тревог сопровождается снижением полноты. Интервалы неопределённости по короткому периоду не заявляются.',small))
    story+=[PageBreak(),p('R13: выбор по данным разработки',heading)]
    second=root/'report/assets/adaptive_development.png';story.append(Image(str(second),width=210*mm,height=210*mm*5.2/8.5))
    story.append(p('Рисунок 24. Заранее ограниченная сетка вариантов на повторно используемых данных разработки. Круги соблюдают FPR до 5% в обоих периодах. Звезда - выбранная стратегия. Октябрьские исходы не участвовали в этом выборе.',small))
    story+=[NextPageTemplate('portrait'),PageBreak(),p('R13: вывод и ограничения',heading),
        p('На полном октябре выбранная обновляемая нейросеть выдала 6 тревог, все ложные: полнота 0 из 77, FPR 1.1%. Статическая нейросеть R12 выдала 96 тревог, из них 15 верных: полнота 19.5%, FPR 14.9%. Сократить ложные срабатывания удалось ценой пропуска всех целевых превышений в выбранной рабочей точке.'),
        p('Обновляемый простой фон выдал 11 тревог, из них 3 верных: точность тревог 27.3%, полнота 3.9%, FPR 1.5%. Заданный ориентир - от 30% верных тревог, полнота от 10%, FPR до 5% и преимущество над простым методом - не выполнен. Результат не используется для дальнейшей настройки по октябрю.'),
        p('Короткий контроль не доказывает невозможность прогнозирования. Полный месяц даёт два полных 14-суточных блока и один короткий; первые 21 сутки - один полный и один короткий. При минимуме четырёх блоков для интервала парной разности Brier доверительные границы остаются пустыми. Полные октябрьские метки предварительные; первые 21 сутки имеют больше времени на регистрацию.'),
        p('Опыт ретроспективный. Исторические даты публикации открытого портала неизвестны; report_date_time не заменяет first-seen. Принятые задержки и созревание меток остаются допущениями. Ни внешнего пилота, ни проспективно выданных октябрьских прогнозов не было.'),
        p('Исследовательский итог: качество ранжирования, калибровка и полезность конкретной тревоги различаются. Для проверенных моделей не подтверждена рабочая точка, одновременно обеспечивающая заданные чувствительность и частоту ошибок. Это ограничение конкретного опыта и доступной информации, а не универсальная граница предсказуемости преступности.'),
        p('Пройдены 118 программных тестов. Отдельно воспроизведены все 27 вариантов разработки и ежемесячные политики, подготовка новых случаев, два файла прогнозов и все итоговые таблицы. Каталог результатов: results/adaptive_seattle_detector. Источник: Seattle SPD Crime Data, tazs-3rd5. Репозиторий: https://github.com/Alexandch/hawkes-criticality-lab.',small)]
    doc.build(story);extra=PdfReader(supplement);writer=PdfWriter();writer.append(original);writer.append(extra,import_outline=False)
    writer.add_outline_item('Дополнение R13 - заключительная попытка, 22.11.2026',offset);writer.add_metadata({'/Title':'Исследование критических режимов - отчёт R1-R13','/Author':'Hawkes Criticality Lab'})
    target=root/'output/pdf/hawkes_criticality_report.pdf'
    with target.open('wb') as handle:writer.write(handle)
    record=dict(base_revision=BASE_REVISION,base_pages=offset,base_sha256=digest(base),supplement_pages=len(extra.pages),total_pages=offset+len(extra.pages),
        output_sha256=digest(target),builder_sha256=digest(Path(__file__)),source_sha256={str(p.relative_to(root)).replace('\\','/'):digest(p) for p in [out/'headline.csv',out/'frozen_plan.json',out/'assessment.json',first,second]})
    (root/'output/pdf/r13_build.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
