"""Append R15 to the preserved62-page R1-R14 report."""
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

BASE_REVISION='35c1882cd5fd2716e34357a3a9246c3e93972c49'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];temporary=root/'tmp/pdfs/r15';temporary.mkdir(parents=True,exist_ok=True)
    base=temporary/'base_r1_r14.pdf';base.write_bytes(subprocess.check_output(['git','show',BASE_REVISION+':output/pdf/hawkes_criticality_report.pdf'],cwd=root))
    original=PdfReader(base);offset=len(original.pages);assert offset==62;out=root/'results/selective_austin_detector'
    with (out/'metrics.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
    bound=json.loads((out/'new_input.json').read_text(encoding='utf-8'))
    for name,file in [('R15Times','times.ttf'),('R15Bold','timesbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('R15Times',normal='R15Times',bold='R15Bold',italic='R15Times',boldItalic='R15Bold')
    body=ParagraphStyle('Body',fontName='R15Times',fontSize=11,leading=15,spaceAfter=10)
    small=ParagraphStyle('Small',parent=body,fontSize=9.5,leading=13,spaceAfter=9)
    heading=ParagraphStyle('Heading',fontName='R15Bold',fontSize=18,leading=23,spaceAfter=15)
    cell=ParagraphStyle('Cell',fontName='R15Times',fontSize=9,leading=12)
    def p(text,style=body):return Paragraph(text,style)
    def table(values,widths):
        result=Table([[p(str(c),cell) for c in row] for row in values],colWidths=widths,repeatRows=1)
        result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#c3cdd6')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));return result
    supplement=temporary/'supplement.pdf';doc=BaseDocTemplate(str(supplement),pagesize=A4,title='Дополнение R15 от 22 ноября 2026',author='Hawkes Criticality Lab')
    def footer(canvas,document):
        w,h=canvas._pagesize;canvas.saveState();canvas.setFont('R15Times',8);canvas.setFillColor(colors.HexColor('#53606b'))
        canvas.drawString(18*mm,12*mm,'Hawkes Criticality Lab - созревание меток и отбор тревог R15, 22.11.2026')
        canvas.drawRightString(w-18*mm,12*mm,str(offset+document.page));canvas.restoreState()
    for name,size in [('portrait',A4),('landscape',landscape(A4))]:
        w,h=size;doc.addPageTemplates(PageTemplate(id=name,pagesize=size,frames=[Frame(18*mm,23*mm,w-36*mm,h-41*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))
    story=[p('Дополнение R15. Созревание меток и избирательные тревоги',heading),
        p('Проверяются недельное обновление, калибровка риска по секторам и отказ от тревоги при недостаточном риске. Сохранены три NB2-сети без погоды из R14, признаки, календарь и цель превышения. Повторного обучения весов нет.'),
        p('Просмотренный Austin 2025 переведён в разработку. Новый контроль - январь-август 2026. Сентябрь-ноябрь исключены до загрузки, чтобы оставить время на регистрацию. Правила и модель зафиксированы до получения новых строк.',small),
        p('Диагностика доступной разметки. Для месячных обновлений 2025 года сопоставлены поздний снимок и метки, известные на дату обновления. Рассмотрены предыдущие 90 суток при выдержке 28/56/90 суток. Часть истинных превышений ещё не достигала порога по доступным записям и считалась отрицательными примерами.',small),
        table([['Выдержка','Превышений в окнах','Ещё не распознано','Доля'],['28 суток','4 652','434','9.3%'],['56 суток','4 656','276','5.9%'],['90 суток','4 662','199','4.3%']],[105,160,140,88]),Spacer(1,12),
        p('Окна перекрываются и содержат повторные события; это описательная диагностика, не независимые испытания. При смене выдержки меняются сами окна. Созревание не восстанавливает никогда не зарегистрированные преступления.',small),
        p('Новое правило. Каждые семь суток используется окно 180 суток, заканчивающееся за 56 суток до обновления. Первые 90 суток калибруют вероятности, вторые - отрицательный квантиль исходного score при уровне 5%. Общий положительный наклон логита дополнен регуляризованными поправками по секторам. Тревога требует также калиброванного риска не ниже 30%. Это правило решения, а не гарантия точности 30%.',small),
        p('Шесть вариантов: NB2 / простой фон с тремя правилами - месячным, недельным с общей калибровкой, недельным по секторам. На каждом квартале 2025 требуются FPR до 5% и полнота от 10%. Ни один кандидат не прошёл; по резервному правилу выбрана недельная NB2 по секторам. Разработка: 42/137 верных тревог, точность 30.7%, полнота 2.8%, FPR 0.8%.',small),
        PageBreak(),p('R15: отдельный контроль 2026 года',heading),
        p('Загружены 19 905 строк; 192 строки с другими или отсутствующими секторами исключены. Получены 19 713 новых случаев. Прежние 169 814 сохранены без изменений; общая панель - 189 527. Контроль: 8 748 ячеек, 935 превышений, 18 временных блоков.',small)]
    values=[['Правило','TP / тревог','Верные тревоги','Полнота','FPR']]
    for role,label in [('monthly_neural','Прежняя месячная NB2'),('selected','Недельная NB2 по секторам'),('baseline','Недельный простой фон')]:
        row=next(r for r in rows if r['scenario']=='primary' and r['group']=='all' and r['role']==role)
        values.append([label,row['true_positives']+' / '+row['alarms'],*[f'{100*float(row[key]):.1f}%' if row[key] else 'не опр.' for key in ('precision','recall','fpr')]])
    story += [table(values,[166,84,91,77,75]),Spacer(1,12),
        p('FPR снизился с 9.9% до 2.2%, полнота - с 18.2% до 5.1%; доля верных тревог выросла с 18.0% до 21.8%. Это сравнение на одном новом периоде, не прямое сравнение с тестом R14 за другой год. Выбранная, нейросетевая и секторная роли совпадают. Метод без тревог имеет неопределённую точность и нулевую полноту.',small),
        p('95% условные интервалы блочного bootstrap: точность 15.8-29.3%, полнота 3.3-7.0%, FPR 1.4-3.1%. В отдельных секторах FPR достигает 9.4%. При лагах доступности и публикации 14 суток: 56/232 верных тревог, точность 24.1%, полнота 6.0%, FPR 2.3%.',small),
        p('AP выбранной сети = 0.1589 против 0.1386 у простого метода, Brier = 0.09582 против 0.09475. Интервал разности Brier [-0.00027; 0.00269] не подтверждает выигрыша. Прежняя месячная NB2 имеет AP = 0.1598, Brier = 0.09489. Новая калибровка не улучшила качество вероятностных оценок.',small),
        p('Совместный ориентир - минимум 30% верных тревог, полнота от 10%, FPR до 5% и преимущество над простым методом - не выполнен. Ни один из 30 зависимых срезов не прошёл фильтр; пять ролей содержат только три уникальные стратегии. Низкая частота ложных тревог достигнута ценой пропуска большинства превышений.',small),
        p('Ограничения. Зарегистрированные письменные отчёты не равны всем преступлениям. Исторические first-seen неизвестны; приняты условные лаги публикации. Поздние метки 2026 и исправления предварительны. Проспективных прогнозов и внешнего пилота не было. Опыт одновременно меняет выдержку, частоту обновления и калибровку, поэтому не изолирует эффект каждого изменения.',small),
        NextPageTemplate('landscape'),PageBreak(),p('R15: рабочая точка на новом периоде',heading)]
    first=root/'report/assets/selective_operating.png';second=root/'report/assets/selective_maturity.png'
    story += [Image(str(first),width=258*mm,height=258*mm*4.4/13),Spacer(1,12),
        p('Рисунок 27. Прежняя и новая рабочие точки на январе-августе 2026. Полосы - 95% условные интервалы блочного bootstrap по 18 временным блокам, 1 000 повторов. FPR использует ячейки без превышения, точность - все тревоги. Для метода без тревог точность не определена.',small),
        p('Более длительное созревание уменьшило неполноту калибровочных меток. Но комбинация более строгих условий не обеспечила одновременно достаточные точность и полноту. Дополнительная настройка по уже просмотренному контролю потребует другого нового периода.',small),
        PageBreak(),p('R15: диагностика неполноты разметки',heading),
        Image(str(second),width=225*mm,height=225*mm*4.2/9),Spacer(1,10),
        p('Рисунок 28. Доля ещё не распознанных превышений в перекрывающихся окнах разработки. Выдержка 90 суток использована только для диагностики; новый алгоритм фиксирует 56 суток. Остаточная поздняя регистрация сохраняется.',small),
        p('Проверка: пройдены 133 программных теста; независимо воспроизведены подготовка новых случаев, все шесть вариантов и квартальный выбор, диагностика, прежние прогнозы R14, оба сценария и все 30 общих, 270 секторных и 240 месячных строк. Материалы: results/selective_austin_detector; scripts/verify_selective_alarm.py.',small),
        p('Источник: Austin Crime Reports, fdj4-gpfu; https://catalog.data.gov/dataset/crime-reports. Репозиторий: https://github.com/Alexandch/hawkes-criticality-lab. Единственный README содержит полный отчёт. Сырые случаи и большие прогнозные массивы остаются внешними.',small)]
    doc.build(story);extra=PdfReader(supplement);assert len(extra.pages)==4
    writer=PdfWriter();writer.append(original);writer.append(extra,import_outline=False);writer.add_outline_item('Дополнение R15 - созревание меток и отбор тревог',offset)
    writer.add_metadata({'/Title':'Исследование критических режимов - отчёт R1-R15','/Author':'Hawkes Criticality Lab'})
    target=root/'output/pdf/hawkes_criticality_report.pdf'
    with target.open('wb') as handle:writer.write(handle)
    record=dict(base_revision=BASE_REVISION,base_pages=offset,base_sha256=digest(base),supplement_pages=len(extra.pages),total_pages=offset+len(extra.pages),
        output_sha256=digest(target),builder_sha256=digest(Path(__file__)),source_sha256={p.relative_to(root).as_posix():digest(p) for p in
        [out/'metrics.csv',out/'new_input.json',out/'frozen_plan.json',out/'assessment.json',out/'verification.json',first,second]})
    (root/'output/pdf/r15_build.json').write_text(json.dumps(record,indent=2),encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
