# Sprint 1 expanded deck: prepared speaker notes

These are the prepared notes, not a transcript or a measured record of delivery.

## Slide 1

Goodfriend (0:00-0:20)
Hello, we are Team WTM: Goodfriend, Morisade, Tari and Tayaphon. We want to predict whether a waterpoint works, needs repair, or does not work. Sprint 1 checked the data and compared pump conditions across different groups. Tayaphon will introduce the dataset.

## Slide 2

Tayaphon (0:20-1:00)
We have 59,400 waterpoint records, collected between 2002 and 2013. We grouped the 40 original input columns into topics so we could ask clear questions about location, pump construction, water conditions and management. The three labels are uneven. About 54 percent work, 38 percent do not work, and only seven percent work but need repair. That small repair group will need separate checks when we build models. First, Morisade will explain the data problems that affect our comparisons.

## Slide 3

Morisade (1:00-1:45)
About 35 percent of construction years are unknown. Four regions have no usable years and contain one fifth of all records. Our age results therefore describe only part of the dataset. We also found 2,145 different installer names, including spelling variants. The amount and height correlation falls from 0.34 to 0.02 when we leave out zero heights. That shows how data choices can change a result. Our shared cleaned copy keeps every row and marks uncertain values. Some questions remain open. Tari will now show the patterns we can describe with those limits.

## Slide 4

Tari (1:45-2:50)
Water availability has a clear link with pump condition. Almost 97 percent of dry waterpoints are not working, compared with about 28 percent where water is recorded as enough. However, dry may describe a problem that already exists. We will compare models with and without this field. Age also shows a difference. Among records with usable ages, about 72 percent of the youngest group work, compared with 31 percent of the oldest. The youngest group also works more often within all 16 regions and five pump groups with enough records. The middle age groups do not always fall steadily. These comparisons do not prove that age or water supply caused the condition. Tayaphon will show what we found in the other categories.

## Slide 5

Tayaphon (2:50-3:50)
The left chart compares how strongly each field is linked to pump status. Water quantity ranks highest among the categorical fields we checked. Pump type and outlet type follow. Permit has a much weaker link on its own, but that does not prove it is useless in a model. The right chart gives three examples. Multiple standpipes have a higher non-working share than single standpipes. Lake sources have a higher share than springs. Waterpoints where users never pay have a higher share than those with annual payments. These groups also differ in location, age and other ways. We cannot say that changing the outlet, source or payment would fix a pump. Goodfriend will explain how these findings guide Sprint 2.

## Slide 6

Goodfriend (3:50-4:30)
We will compare simple and more complex models using the same data preparation. We will test water quantity both ways, check repeated information, and review installer name variants. We will learn preprocessing choices from training data only. Always predicting working would already score 54 percent overall, while missing every repair case. So we will report precision, recall and F1 for each pump state. We will check missed repairs and false alarms. That will show whether our models improve on a simple starting point. Thank you.
