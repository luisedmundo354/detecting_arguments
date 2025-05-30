import graphviz
import textwrap

dot = graphviz.Digraph(comment='Extended Syllogism Tree')
dot.attr(size="8.5,11!", page="8.5,11")
dot.attr(rankdir="BT")
dot.attr('node', shape='none')

def html_label(title, content, width=150, wrap_width=40):
    if title.startswith("Major Premise"):
        title = title.replace("Major Premise", "Rule", 1)
    elif title.startswith("Minor Premise"):
        title = title.replace("Minor Premise", "Analysis", 1)
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

# Level 
node_def_Ma1 = html_label(
    "Major Premise (Binding Authority)",
    "Generally, the cost basis of property purchased with other property is the fair market value of the property received in the exchange."
)
node_def_Mi1 = html_label(
    "Minor Premise",
    "The general rule should be applied to the case."
)
node_def_Ma2 = html_label(
    "Major Premise",
    "The cost basis Ferroxcube must use is the fair market value of the property"
)

dot.node('dMa1', label=node_def_Ma1)
dot.node('dMi1', label=node_def_Mi1)
dot.node('dMa2', label=node_def_Ma2)

dot.edge('dMa1', 'dMa2', constraint='true')
dot.edge('dMi1', 'dMa2', constraint='true')
dot.edge('dMa1', 'dMi1', style='dashed', arrowhead='none', constraint='false')

# Level 
node_def_Ma31 = html_label(
    "Major Premise (Binding Authority)",
    "Where stock is exchanged for property pursuant to an arm’s-length transaction, the courts have, in certain instances, presumed that the value of such stock equaled the value of the  property received in exchange therefor. Pittsburgh Terminal Corp. This is sometimes referred to as the barter-equation method of valuation."
)
node_def_Mi31 = html_label(
    "Minor Premise",
    "Under these circumstances, we cannot find that petitioner and Ferroxcube were dealing at arm’s length in the ordinary sense of that term."
)
node_def_Mi32 = html_label(
    "Minor Premise",
    "The court cannot presume that the value of the stock equale the value of the property."
)

dot.node('dMa31', label=node_def_Ma31)
dot.node('dMi31', label=node_def_Mi31)
dot.node('dMi32', label=node_def_Mi32)

dot.edge('dMa31', 'dMi32', constraint='true')
dot.edge('dMi31', 'dMi32', constraint='true')
dot.edge('dMa31', 'dMi31', style='dashed', arrowhead='none', constraint='false')

# Level 
node_def_Ma41 = html_label(
    "Major Premise (genral rule)",
    "Implicit: In cases where the best alternative to a negotiated agreement is significantly worse, the seller is in disadvantage."
)
node_def_Mi41 = html_label(
    "Minor Premise (Evidence)",
    "It appears that *965 petitioner was the only party interested in purchasing the Memory Systems Division,5 leaving Ferroxcube with the choice of selling the division to petitioner or  scrapping its assets for salvage value."
)
node_def_Mi42 = html_label(
    "Major Premise",
    "It is clear that Ferroxcube was placed at a distinct disadvantage in negotiating a sale to petitioner."
)

dot.node('dMa41', label=node_def_Ma41)
dot.node('dMi41', label=node_def_Mi41)
dot.node('dMa42', label=node_def_Mi42)

dot.edge('dMa41', 'dMa42', constraint='true')
dot.edge('dMi41', 'dMa42', constraint='true')
dot.edge('dMa41', 'dMi41', style='dashed', arrowhead='none', constraint='false')

# Level 
node_def_Ma3 = html_label(
    "Major Premise (rule)",
    "Arm's length is commonly used to refer to transactions in which two or more unrelated and unaffiliated parties agree to do business, acting independently and in their self-interest."
)
node_def_Mi3 = html_label(
    "Minor Premise",
    "The evidence shows that Ferroxcube was not acting independently."
)

dot.node('dMa3', label=node_def_Ma3)
dot.node('dMi3', label=node_def_Mi3)

dot.edge('dMa3', 'dMi31', constraint='true')
dot.edge('dMi3', 'dMi31', constraint='true')
dot.edge('dMa3', 'dMi3', style='dashed', arrowhead='none', constraint='false')

# level
node_def_Ma51 = html_label(
    "Major Premise (rule)",
    "Fair market value has been defined as the price at which property would be sold by a knowledgeable seller to a knowledgeable buyer with neither party under any compulsion to act."
)
node_def_Mi51 = html_label(
    "Minor Premise",
    "We believe, in light of the lack of any evidence to the contrary, that respondent acted reasonably in assuming that Ferroxcube found itself forced to sell the Memory Systems Division’s assets to  petitioner at a price below their fair market value rather than scrapping those assets for salvage value."
)
node_def_Mi52 = html_label(
    "Minor Premise",
    "Ferroxcube was not using fair market value."
)

dot.node('dMa51', label=node_def_Ma51)
dot.node('dMi51', label=node_def_Mi51)
dot.node('dMa52', label=node_def_Mi52)

dot.edge('dMa51', 'dMa52', constraint='true')
dot.edge('dMi51', 'dMa52', constraint='true')
dot.edge('dMa51', 'dMi51', style='dashed', arrowhead='none', constraint='false')

dot.edge('dMa52', 'dMi3', constraint='true')
dot.edge('dMa42', 'dMi3', constraint='true')



# Below is structure for distinguishable claim
node_Ma = html_label(
    "Major Premise (Binding Authority)",
    "It is nonsensical to consider stock value separately from property value (Implicit: both stock and property are necessary for proper valuation)."
)
node_Mi = html_label(
    "Minor Premise (Evidence)",
    "In Pittsburgh, the stock value was determined based on the property's market value (both stock and property values were considered)."
)
node_Ma1 = html_label(
    "Major Premise",
    "We have always followed the most reliable evidence of value from both sides of the transaction."
)

dot.node('Ma', label=node_Ma)
dot.node('Mi', label=node_Mi)
dot.node('Ma1', label=node_Ma1)

dot.edge('Ma', 'Ma1', constraint='true')
dot.edge('Mi', 'Ma1', constraint='true')
dot.edge('Ma', 'Mi', style='dashed', arrowhead='none', constraint='false')

# Level 2: Intermediate branches

## Branch A: Evidence on property price reliability
# Note: There is a conflict here – some evidence suggests that property price was not chosen as the reliable indicator,
# while other premises state that it was chosen. This contradiction is highlighted below.
node_Mi11 = html_label(
    "Minor Premise",
    "For Ferroxcube, the court followed the evidence and did not choose the property price as the most reliable indicator."
)
node_Mi111 = html_label(
    "Minor Premise",
    "For Ferroxcube, the property price is not the most reliable evidence of the value."
)
node_Mi12 = html_label(
    "Minor Premise",
    "In Pittsburgh, the court followed the evidence and chose the property price as the most reliable indicator."
)
node_Mi121 = html_label(
    "Minor Premise",
    "In Pittsburgh, the property price is the most reliable evidence of the value."
)
node_Ma3 = html_label(
    "Major Premise (general rule)",
    "Implicit: If the price base of two cases is not the same, they are different ragarding which side of the transaction value is more reliable."
)

node_Mi13 = html_label(
    "Minor Premise",
    "Pittsburgh and Ferroxcube differ regarding which side of the transaction value is deemed more reliable."
)

dot.node('Mi11', label=node_Mi11)
dot.node('Mi111', label=node_Mi111)
dot.node('Mi12', label=node_Mi12)
dot.node('Mi121', label=node_Mi121)
dot.node('Ma3', label=node_Ma3)
dot.node('Mi13', label=node_Mi13)

dot.edge('Mi11', 'Mi111', constraint='true')
dot.edge('Mi12', 'Mi121', constraint='true')
dot.edge('Ma1', 'Mi111', constraint='true')
dot.edge('Ma1', 'Mi121', constraint='true')
dot.edge('Ma1', 'Mi11', style='dashed', arrowhead='none', constraint='false')
dot.edge('Ma1', 'Mi12', style='dashed', arrowhead='none', constraint='false')

dot.edge('Ma3', 'Mi13', constraint='true')
dot.edge('Mi111', 'Mi13', constraint='true')
dot.edge('Mi121', 'Mi13', constraint='true')
dot.edge('Ma3', 'Mi111', style='dashed', arrowhead='none', constraint='false')
dot.edge('Ma3', 'Mi121', style='dashed', arrowhead='none', constraint='false')

## Branch B: Evidence on stock-for-assets transactions
node_stock_pitt = html_label(
    "Major Premise (genral rule)",
    "If the proportions of assets and stock transacted by the company in the precedent case is different, then the cases differ in the proportion of stock and assets transacted."
)
node_stock_ferro = html_label(
    "Minor Premise (Evidence)",
    "In Pittsburgh, almost all stock was exchanged for assets that comprised nearly all of the company's assets. In Ferroxcube, only a small fraction of stock was exchanged for assets representing a fraction of the total assets."
)
node_stock_diff = html_label(
    "Minor Premise",
    "Pittsburgh and Ferroxcube differ in the proportion of stock sold and the corresponding fraction of assets transacted."
)

dot.node('stock_pitt', label=node_stock_pitt)
dot.node('stock_ferro', label=node_stock_ferro)
dot.node('stock_diff', label=node_stock_diff)

dot.edge('stock_pitt', 'stock_diff', constraint='true')
dot.edge('stock_ferro', 'stock_diff', constraint='true')
dot.edge('stock_ferro', 'stock_pitt', style='dashed', arrowhead='none', constraint='false')

# Level 3: Final syllogism regarding distinguishability and binding nature
node_if_diff = html_label(
    "Major Premise (Implicit)",
    "If there is evidence of differences between Pittsburgh and Ferroxcube, then they are distinguishable."
)
node_dist_concl = html_label(
    "Minor Premise",
    "Based on the record, Pittsburgh Terminal Corp. is distinguishable."
)
node_ruling_not_bind = html_label(
    "Major Premise (Implicit)",
    "The ruling is not binding in the case at hand, if the precendent is distinguishable."
)
node_final = html_label(
    "Minor Premise",
    "The ruling is not binding in the current case because the precedent is distinguishable."
)

dot.node('if_diff', label=node_if_diff)
dot.node('dist_concl', label=node_dist_concl)
dot.node('ruling_not_bind', label=node_ruling_not_bind)
dot.node('final', label=node_final)

dot.edge('stock_diff', 'dist_concl', constraint='true')
dot.edge('Mi13', 'dist_concl', constraint='true')
dot.edge('if_diff', 'dist_concl', constraint='true')
dot.edge('stock_diff', 'if_diff', style='dashed', arrowhead='none', constraint='false')
dot.edge('Mi13', 'if_diff', style='dashed', arrowhead='none', constraint='false')

dot.edge('ruling_not_bind', 'final', constraint='true')
dot.edge('dist_concl', 'final', constraint='true')
dot.edge('ruling_not_bind', 'dist_concl', style='dashed', arrowhead='none', constraint='false')



# Closing Argument
node_def_Ma6 = html_label(
    "Major Premise (rule)",
    "Implicit: When a precedent is not binding, a more general rule must be applied."
)

dot.node('dMa6', label=node_def_Ma6)

dot.edge('dMa6', 'dMi1', constraint='true')
dot.edge('final', 'dMi1', constraint='true')
dot.edge('dMa6', 'final', style='dashed', arrowhead='none', constraint='false')


node_def_Ma7 = html_label(
    "Conclusion",
    "Since we have rejected the use of this method of valuation in the instant case, we find that petitioner has failed to meet its burden of proof and sustain respondent’s determination."
)

dot.node('dMa7', label=node_def_Ma7)

dot.edge('dMa2', 'dMa7', constraint='true')
dot.edge('dMi32', 'dMa7', constraint='true')
dot.edge('dMa2', 'dMi32', style='dashed', arrowhead='none', constraint='false')


print(dot.source)
dot.render('extended_tree_structure', view=True)