# Sprint 1 presented deck: prepared speaker notes

These are the prepared notes, not a transcript or a measured record of delivery.

## Slide 1

Goodfriend (0:00-0:25)
Hello, we are Team WTM: Goodfriend, Morisade, Tari and Tayaphon. Our project looks at water pumps in Tanzania. In Sprint 1, we checked the data and studied which features are linked to pumps working or needing repair. Tayaphon will begin with what the dataset contains.

## Slide 2

Tayaphon (0:25-1:10)
We have 59,400 waterpoint records and 40 original input columns. The bars group these columns by topic. They cover location and population, construction and pump type, water conditions, management and payment, and record details. One field still has an unclear meaning. The records were entered between 2002 and 2013. Each waterpoint has one of three labels: working, needing repair while still working, or not working. We compare these topics with the three pump states. Before making those comparisons, Morisade will explain where the data is missing.

## Slide 3

Morisade (1:10-2:00)
About 35 percent of construction years are unknown. The missing share is higher among waterpoints needing repair. Four regions have no recorded construction years at all: Dodoma, Kagera, Mbeya and Tabora. They account for about one fifth of all records, so age findings leave out a large part of the dataset. We have saved a shared cleaned copy. It keeps every record and marks missing values and impossible ages. Repeated records remain for review. The same code lets everyone repeat these steps. With those limits clear, Tari will show the main patterns.

## Slide 4

Tari (2:00-3:05)
First, water supply. Almost 97 percent of waterpoints recorded as dry are not working. Where water is recorded as enough, about 65 percent are working. The age chart also shows a clear difference. Among records with usable ages, the working share falls from about 72 percent in the youngest group to 31 percent in the oldest. We checked the youngest and oldest groups within regions and pump technologies too. The younger group works more often in all 16 regions and five pump groups with enough records. The middle age bands do not always fall steadily. Other factors may explain these differences. The comparisons cannot tell us what caused a pump to stop working. Tayaphon will now show a check on the numeric fields.

## Slide 5

Tayaphon (3:05-3:50)
These tables compare four numeric fields. A positive value means higher values in one field tend to go with higher values in the other. The amount and height link is 0.34 when zero heights are kept. It falls to 0.02 when they are left out. Zero height may sometimes mean missing data, but we have not confirmed every case. We also need to confirm what the amount field measures. We keep the original values and record these questions. In Sprint 2, we will test how these choices change model results. Goodfriend will explain what we will measure.

## Slide 6

Goodfriend (3:50-4:30)
Sprint 2 will compare a simple starting model with more complex models. We will use the shared cleaning steps. We will choose missing-value fills using the training data only. We will check results for each pump state. Only 4,317 waterpoints need repair, about seven percent of the dataset. A single accuracy score could hide poor results for this group. We will count missed repairs and false alarms as well as correct predictions. That is how we will judge whether the models are useful. Thank you.
