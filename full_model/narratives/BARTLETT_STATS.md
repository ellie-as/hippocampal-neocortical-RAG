# Bartlett / Bergman-Roediger Stats

Figure 5b replots Bergman and Roediger (1999), using the experimental
condition subset with `n=8` participants who completed all three recalls.
The relevant source values are from Tables 3 and 4.

The paper reports mean proportions out of 42 possible propositions:

```text
15 min:   0.19 accurate + 0.21 minor + 0.15 major = 0.55
1 week:   0.09 accurate + 0.18 minor + 0.18 major = 0.45
6 months: 0.04 accurate + 0.07 minor + 0.16 major = 0.27
```

For Figure 5b, these are normalized within each delay to show fractions of
recalled propositions. Omitted propositions are therefore excluded:

```text
15 min:   0.345 no + 0.382 minor + 0.273 major = 1.000
1 week:   0.200 no + 0.400 minor + 0.400 major = 1.000
6 months: 0.148 no + 0.259 minor + 0.593 major = 1.000
```

The original paper reports SDs and does not plot error bars. If error bars are
shown here, they should be SEMs derived from the reported SDs using
`SEM = SD / sqrt(8)`, then normalized to the same fraction-of-recalled scale.
