# Product

## Register

product

## Users

Des équipes bancaires côté relation client / rétention (chargés de clientèle, analystes marketing) qui veulent anticiper le départ ("attrition") des porteurs de carte de crédit. Elles utilisent l'outil de deux façons : au cas par cas, en saisissant les données d'un client dans un formulaire ; ou en lot, en tirant un échantillon de clients depuis la base pour scanner rapidement un portefeuille.

## Product Purpose

Réduire le taux d'attrition des clients carte de crédit en donnant aux équipes un signal de risque immédiat et actionnable : à partir de quelques variables client (âge, ancienneté produits, inactivité, solde, transactions, taux d'utilisation), un modèle de classification (scikit-learn) prédit si le client est sur le point de résilier ou non. Le succès se mesure à la capacité de l'outil à être consulté et compris en quelques secondes par quelqu'un qui n'est pas data scientist.

## Brand Personality

Sobre, rigoureux, rassurant — direction "corporate bancaire clair" : fond clair, palette sobre (bleu/marine ou équivalent) avec un accent de confiance, sérieux sans être froid ni austère.

## Anti-references

- Le look "admin template" générique (Bootstrap par défaut, dashboard interchangeable qu'on retrouve partout) — explicitement à éviter.
- L'état actuel du site lui-même est un anti-pattern à corriger : trois systèmes visuels incompatibles cohabitent (thème HTML5UP sur l'accueil/BD, thème "carte" façon SaaS sur le formulaire, Bootstrap CDN + styles inline sur les pages de résultat), sans cohérence de typographie, de couleur ni d'espacement.
- Bug de lisibilité critique sur les pages de résultat (`Answers0.html` / `Answers1.html`) : le titre `h1` est de la même couleur que le fond (`cornflowerblue` sur `cornflowerblue`), donc invisible.

## Design Principles

1. **Clarté avant décor** — chaque page sert une tâche précise (saisir, lire un résultat, scanner une liste) ; la mise en page doit rendre cette tâche évidente avant d'être belle.
2. **Un seul système, pas trois** — unifier palette, typographie, composants (boutons, champs, cartes, tableau) sur les 5 pages (`index`, `Predic_form`, `Predic_BD`, `Answers0`, `Answers1`).
3. **Le verdict doit se lire d'un coup d'œil** — les pages de résultat (client va partir / ne va pas partir) doivent communiquer l'information sans ambiguïté, y compris sans percevoir la couleur (icône/texte en plus de la couleur).
4. **Confiance visuelle** — palette et typographie sobres, professionnelles, qui inspirent la confiance d'un outil bancaire réel, pas d'un prototype.
5. **Accessible par défaut** — contrastes AA partout, navigation clavier, `prefers-reduced-motion` respecté pour toute animation ajoutée.

## Accessibility & Inclusion

WCAG AA : contraste texte ≥4.5:1 (≥3:1 pour les grands textes), focus clavier visibles sur tous les éléments interactifs (formulaire, boutons, liens de menu), alternative non-couleur pour le statut "client à risque / client stable" (icône ou libellé, pas seulement rouge/vert), et respect de `prefers-reduced-motion` pour toutes les animations.
