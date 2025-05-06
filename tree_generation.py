import graphviz
import textwrap

dot = graphviz.Digraph(comment='Syllogism')
dot.attr(size="8.5,11!", page="8.5,11")
dot.attr('node', shape='none')

def html_label(title, content, width=150, wrap_width=40):

    wrapped_content = textwrap.fill(content, wrap_width)

    wrapped_content_html = wrapped_content.replace("\n", "<BR ALIGN='left'/>")
    return f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">
  <TR>
    <TD BGCOLOR="lightblue" WIDTH="{width}"><B>{title}</B></TD>
  </TR>
  <TR>
    <TD ALIGN="left" WIDTH="{width}">{wrapped_content_html}</TD>
  </TR>
</TABLE>
>'''


long_text_Ma = "While it may be true that the stock of a going concern can be valued *964 independently of the assets which are acquired with such stock, we think that it is nonsensical to try to look at the value of the stock separately from that of the property where, as here, all of the stock of a corporation which has never conducted any business is exchanged for property which will become all of the assets of such corporation."
long_text_Mi = "In Pittsburgh Terminal Corp., a newly formed corporation received virtually all of its assets in exchange for all but an insignificant portion of its stock. To ascertain the corporation’s basis in the property received, this Court determined the value of such stock by reference to the fair market value"
long_text_C = "Any evaluation of the stock of the corporation will necessarily be based upon the income which can be expected from the property.   Regardless of the precise rule of valuation that we were attempting to apply, we have always followed the evidence of value on either side of the transaction which we considered to be the most reliable. * * * [60 T.C. at 88; emphasis added."

long_text_Mi11 = "We do not consider the fair market value of the property petitioner received from Ferroxcube as reliable evidence of the value of the preferred shares issued to Ferroxcube. In Pittsburgh Terminal Corp., we emphasized that reliable evidence of value on either side of a transaction is crucial, and there, the property's value received in exchange for the corporation's stock was deemed reliable evidence of the stock's value."
long_text_Mi12 = "First, in Pittsburgh Terminal Corp., all but an insignificant portion of the corporation’s stock was given as consideration for the property, and such property thereafter represented all of the corporation’s assets. Therefore, the value of the corporation’s stock necessarily reflected the value of the underlying corporate assets."
long_text_C1 = "On the basis of the record before us, we find that Pittsburgh Terminal Corp. is distinguishable."


node_Ma_label = html_label(
    "Major Premise(grounded on binding authority)", 
    long_text_Ma
    )
node_Mi_label = html_label(
    "Minor Premise(grounded on evidence)", 
    long_text_Mi    
    )
node_C_label = html_label(
    "Conclusion/Major Premise", 
    long_text_C
    )

node_Mi11_label = html_label(
    "Minor Premise(grounded on evidence)", 
    long_text_Mi11
    )
node_Mi12_label = html_label(
    "Minor Premise(grounded by evidence)", 
    long_text_Mi12    
    )
node_C1_label = html_label(
    "Conclusion/Minor Premise", 
    long_text_C1
    )

dot.node('Ma', label=node_Ma_label)
dot.node('Mi', label=node_Mi_label)
dot.node('C', label=node_C_label)

dot.node('Mi11', label=node_Mi11_label)
dot.node('Mi12', label=node_Mi12_label)
dot.node('C1', label=node_C1_label)

# dot.edges([('Ma', 'Mi'), ('Mi', 'C')])
dot.edge('Mi', 'C', constraint='true')
dot.edge('Ma', 'C', constraint='true')
dot.edge('Ma','Mi', style='dashed', arrowhead='none', constraint='false')

dot.edge('Mi11', 'C1', constraint='true')
dot.edge('Mi12', 'C1', constraint='true')
dot.edge('C','C1', constraint='true')
dot.edge('Mi11','Mi12', style='dashed', arrowhead='none', constraint='false')

print(dot.source)
dot.render('tree structure', view=True)
