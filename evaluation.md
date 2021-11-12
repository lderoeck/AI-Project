## Recommender evaluation
First, we generate recommendations based on a user's reviews, as outlined above. Then, we evaluate them both quantitatively and qualitatively.

### Quantitative evaluation
For quantitative evaluation, we use the item inventories of users as the ground truth. Note that reviewed items are excluded from the ground truth in order to ensure that training and evaluation data does not overlap.

We use the following evaluation metrics:

#### Evaluation metrics:
- recall@k: # relevant recommendations / # items
- nDCG@k (discounted cumulative gain): DCG / ideal DCG where `DCG = sum[2^{rel_i}-1 / log_2(i+1)]` and `ideal DCG = sum(1/log_2(i+1))`

We then perform an evaluation for all different combination of recommender techniques that were outlined before:
| recall@10 | Euclidian  | Manhattan | Cosine | Cosine + feedback weighting |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| No tf-idf | 0.007783  | 0.007798 | 0.018838 | 0.017088 |
| Tf-idf | 0.011887  | 0.013003 | 0.017975 | 0.016515 |
| Smooth | 0.011895  | 0.012999 | 0.017978 | 0.016513 |
| Sublinear | 0.011887  | 0.013003 | 0.017975 | 0.016515 |
| Smooth Sublinear | 0.011895  | 0.012999 | 0.017978 | 0.016513 |

| nDCG@10 | Euclidian  | Manhattan | Cosine | Cosine + feedback weighting |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| No tf-idf | 0.063212  | 0.063262 | 0.152079 | 0.137673 |
| Tf-idf | 0.081598  | 0.094712 | 0.141916 | 0.129756 |
| Smooth | 0.081671  | 0.094734 | 0.141907 | 0.129739 |
| Sublinear | 0.081598  | 0.094712 | 0.141916 | 0.129756 |
| Smooth Sublinear | 0.081671  | 0.094734 | 0.141907 | 0.129739 |

We see that cosine distance performs best quantitatively, which confirms the hypothesis that games are similar to documents (and thus the problem of predicting recommendations is similar to a ranked search in information retrieval).

For some reason, tf-idf performs slightly worse quantitatively for cosine distance, which was unexpected. Therefore, we will take a closer look at this during qualitative evaluation.

Finally, feedback weighting results in worse performance, which confirms our hypothesis that negative reviews also provide an indication for the user's interest (since he/she had to buy the game in the first place). We further investigate this in qualitative evaluation.

### Qualitative evaluation:
In the qualitative evaluation we manually go over a subset of the recommendations, to evaluate how likely and accurate the recommendations seem for the given users. This step is meant to augment the quantitative evaluation. We hope to spot obvious shortcomings, patterns or bias that might occur in the recommender, since these defects are not observable in the quantitative evaluation. By doing this sanity check we can be sure that the recommender is not failing in any obvious ways.

The following is a small sample of recommendations for the combination of methods with the best performance (cosine distance / no tf-idf / no feedback weighting):
| Reviews  | Recommendations |
| ------------- | ------------- |
| ['Killing Floor']  | ['Killing Floor 2', 'Left 4 Dead 2', 'Call of Duty: World at War', 'Resident Evil Revelations / Biohazard Revelations', 'Left 4 Dead', 'Codename CURE', 'Contagion', 'Piercing Blow', 'Final DOOM', 'Deadfall Adventures']  |
| ['Galactic Civilizations III']  | ['Galactic Civilizations III: Crusade Expansion Pack', 'Galactic Civilizations III - Mercenaries Expansion Pack', 'Endless SpaceÂ® 2', 'Endless SpaceÂ® - Collection', "Sid Meier's CivilizationÂ®: Beyond Earthâ„¢", 'Horizon', 'Pandora: First Contact', 'Stellar Monarch', 'Distant Worlds: Universe', 'Stellaris']  |
| ['Grand Theft Auto V', 'Goat Simulator', 'PAYDAY 2']  | ['Just Causeâ„¢ 3', 'Saints Row 2', 'Just Cause 2', 'Turbo Dismountâ„¢', 'Blockland', 'Grand Theft Auto: Episodes from Liberty City', 'Grand Theft Auto: San Andreas', 'SimplePlanes', 'Saints Row: The Third', "Garry's Mod"]  |
| ['Stardew Valley', 'How to Survive', 'Counter-Strike: Global Offensive', 'Undertale', "Assassin's Creed 2 Deluxe Edition"] | ['Feel The Snow', 'Dead Rising 3 Apocalypse Edition', 'Crea', 'Starbound', "Assassin's CreedÂ® Unity", 'Terraria', "Don't Starve Together", 'Serious Sam 2', 'Dead RisingÂ® 2', 'Scrap Mechanic'] |

We see that for small review sets we still achieve very relevant results. 'Killing floor' is a zombie survival game and as a result the recommender system provides very similar zombie survival games.
Furthermore, recommendations improve with larger review sets. The final example in the sample contains games that are very distinct, such as 'Stardew Valley', 'How to Survive' and 'Assassin's Creed 2 Deluxe Edition'. The recommendations contains another game from the Assassin's Creed franchise, as well as games that contain elements of those reviewed by the user. For example, 'Terraria' is a game that contains the pixel platformer elements from 'Stardew Valley', as well as the survival aspects from 'How to Survive'.

Let's take a look at why the use of feedback weighting might result in worse recommendations:
| Reviews  | Recommendations |
| ------------- | ------------- |
| ['Killing Floor']  | ['Killing Floor 2', 'Left 4 Dead 2', 'Call of Duty: World at War', 'Resident Evil Revelations / Biohazard Revelations', 'Left 4 Dead', 'Codename CURE', 'Contagion', 'Piercing Blow', 'Final DOOM', 'Deadfall Adventures']  |
| ['Galactic Civilizations III']  | ['Galactic Civilizations III: Crusade Expansion Pack', 'Galactic Civilizations III - Mercenaries Expansion Pack', 'Endless SpaceÂ® 2', 'Endless SpaceÂ® - Collection', "Sid Meier's CivilizationÂ®: Beyond Earthâ„¢", 'Horizon', 'Pandora: First Contact', 'Stellar Monarch', 'Distant Worlds: Universe', 'Stellaris']  |
| ['Grand Theft Auto V', 'Goat Simulator', 'PAYDAY 2']  | ['BalanCity', 'Demolition Inc.', 'Fly in the House', 'Learn to Fly 3', 'Crankies Workshop: Grizzbot Assembly', 'Trivia Vault: Mini Mixed Trivia', 'Trivia Vault: Mini Mixed Trivia 2', "Trivia Vault: 1980's Trivia 2", 'Mission: Wolf', 'World of Cinema - Directors Cut']  |
| ['Stardew Valley', 'How to Survive', 'Counter-Strike: Global Offensive', 'Undertale', "Assassin's Creed 2 Deluxe Edition"] | ['Feel The Snow', 'Dead Rising 3 Apocalypse Edition', 'Crea', 'Starbound', "Assassin's CreedÂ® Unity", 'Terraria', "Don't Starve Together", 'Serious Sam 2', 'Dead RisingÂ® 2', 'Scrap Mechanic'] |

We see that the third user's recommendations have changed, meaning that he/she disliked one of the reviewed games. The recommender system therefore tries to recommend games that do not include any of the tags associated with that game. We see that this results in recommendations which are not related to the reviews at all. Moreover, the recommendations contain games that are very niche and unpopular, making it even less likely that the user is interested in them.

Moreover, we didn't expect tfidf weighting to result in worse quantitative performance. So let's see how recommendations have changed with smooth tfidf:
| Reviews  | Recommendations |
| ------------- | ------------- |
| ['Killing Floor']  | ['Killing Floor 2', 'Left 4 Dead 2', 'Damned', 'Left 4 Dead', 'Zombie Panic! Source', 'Contagion', 'Call of Duty: World at War', 'Codename CURE', 'Arizona Sunshine', 'Resident Evil Revelations / Biohazard Revelations']  |
| ['Galactic Civilizations III']  | ['Endless SpaceÂ® 2', "Sid Meier's CivilizationÂ®: Beyond Earthâ„¢", 'Pandora: First Contact', 'Galactic Civilizations III: Crusade Expansion Pack', 'Endless Legendâ„¢', 'Galactic Civilizations III - Mercenaries Expansion Pack', 'Sword of the Stars II: Enhanced Edition', 'Stellar Monarch', 'Stellaris', 'Horizon']  |
| ['Grand Theft Auto V', 'Goat Simulator', 'PAYDAY 2']  | ['Just Causeâ„¢ 3', 'Grand Theft Auto: San Andreas', 'Grand Theft Auto: Episodes from Liberty City', 'Just Cause 2', 'Saints Row 2', 'Turbo Dismountâ„¢', 'Amazing Frog?', 'Saints Row: The Third', 'Goat Simulator: PAYDAY', 'PAYDAYâ„¢ The Heist']  |
| ['Stardew Valley', 'How to Survive', 'Counter-Strike: Global Offensive', 'Undertale', "Assassin's Creed 2 Deluxe Edition"] | ['Dead Rising 3 Apocalypse Edition', 'Dead Rising 2: Off the Record', "Assassin's CreedÂ® Unity", "Don't Starve Together", 'Feel The Snow', 'Dead RisingÂ® 2', 'Dying Light', 'Goat Simulator: GoatZ', 'Terraria', 'Crea'] |

We find that qualitative results have actually somewhat improved. The tfidf weighting mostly had an impact on the order of the recommendations, as is expected when certain features get higher weights than others. We see that this has a large impact for review sets which contain games with common tags, such as those of the third user. The recommender still finds the main open-world games as recommendations, but also finds the first instalment in the 'PAYDAY' series, as well as a cross-over game called 'Goat Simulator: PAYDAY'. This is probably because the tags of 'Goat Simulator' received less weight due to the abundance of simulation games on steam (thus simulation-related tags are common as well).
