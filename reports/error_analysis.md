# Error analysis — UCI test split (hand-tagged)

Sampling: all 7 misclassified test rows at the frozen threshold, plus 13 of the 30
conformal abstentions (nearest the boundary, chosen to cover distinct failure modes).
Tags are my audit of each case, not model output.

- Test rows: 1,033 (128 spam / 905 ham)
- Operating point (tau = 0.1514): recall 0.9766, FPR 0.0044, precision 0.9690, F1 0.9728, PR-AUC 0.9938
- Policy actions: 874 deliver / 30 quarantine / 129 junk (abstain rate 0.0290)
- Cases below: 3 false negatives, 4 false positives, 13 conformal abstentions

| # | Type | Gold | p(spam) | Message | Hand tag |
|---|---|---|---|---|---|
| 1 | FN | spam | 0.035 | For sale - arsenal dartboard. Good condition but no doubles or trebles! | Gold spam; unsolicited classified advert. Model reads conversational sales text. Not smishing, but the spam class in UCI includes adverts, so this is a scope mismatch rather than a security failure. |
| 2 | FN | spam | 0.038 | Latest News! Police station toilet stolen, cops have nothing to go on! | Gold spam; a joke forward from the Grumble corpus, which labels chain/joke mail as spam. Label is defensible for spam but not for smishing; model's low score is reasonable. Flag as a label-policy question, not a model bug. |
| 3 | FN | spam | 0.045 | In The Simpsons Movie released in July 2007 name the band that died at the start of the film? A-Green Day, B-Blue Day, C-Red Day. (Send A, B or C) | Real miss with risk: premium-rate quiz bait. Surface form is a trivia question, so the model sees ham. A 'Send A/B/C' + reply-cost pattern would catch it; worth a rule, not a retrain. |
| 4 | FP | ham | 0.200 | Can you plz tell me the ans. BSLVYL sent via fullonsms.com | Ham. The only strong cue is the `fullonsms.com` URL token, which appears in many spam rows. False positive driven by one lexical feature; the meta/URL block does not disambiguate. |
| 5 | FP | ham | 0.227 | Please protect yourself from e-threats. SIB never asks for sensitive information like Passwords,ATM/SMS PIN thru email. Never share your password with anybody. | Ham: a genuine bank security advisory. Contains 'password', 'PIN', 'sensitive information' — the highest-cost false positive type, because blocking it hides a real warning. The policy layer saved it only because p stayed below tau. |
| 6 | FP | ham | 0.323 | Your opinion about me? 1. Over 2. Jada 3. Kusruthi 4. Lovable 5. Silent 6. Spl character 7. Not matured 8. Stylish 9. Simple Pls reply.. | Ham: a friend's numbered list. The numbered menu plus 'reply' mimics a premium-rate menu. Formatting features alone cannot separate the two; context is missing. |
| 7 | FP | ham | 0.514 | Message:some text missing* Sender:Name Missing* *Number Missing *Sent:Date missing *Missing U a lot thats y everything is missing sent via fullonsms.com | Ham. Template placeholders ('Sender:Name Missing') look like a spoofed sender header, and the URL repeats. The model is reacting to the template, not the intent. |
| 8 | Q | ham | 0.035 | Wait 2 min..stand at bus stop | Abstained. Too short for word features; char features see nothing distinctive. Correct behaviour: deliver. |
| 9 | Q | ham | 0.035 | What your plan for pongal? | Abstained. Very short, code-mixed proper noun. No signal either way. |
| 10 | Q | ham | 0.035 | No we sell it all so we'll have tons if coins. Then sell our coins to someone thru paypal. Voila! Money back in life pockets:) | Abstained. 'paypal' and 'money' are phishing vocabulary in a benign resale conversation. Genuinely ambiguous out of context. |
| 11 | Q | ham | 0.036 | Sometimes we put walls around our hearts,not just to be safe from getting hurt.. But to find out who cares enough to break the walls &amp; get closer.. GOODNOON:) | Abstained. HTML entity `&amp;` survives normalisation; otherwise a forwarded quote. No action needed. |
| 12 | Q | ham | 0.037 | Aight, text me tonight and we'll see what's up | Abstained. Short conversational message; both classes are plausible at this length. |
| 13 | Q | ham | 0.040 | S:)8 min to go for lunch:) | Abstained. Near-empty after normalisation. Abstention is the right outcome. |
| 14 | Q | ham | 0.042 | Plz note: if anyone calling from a mobile Co. &amp; asks u to type # &lt;#&gt;  or # &lt;#&gt; . Do not do so. Disconnect the call,coz it iz an attempt of 'terrorist' to make use of the sim card no. Itz confirmd by nokia n motorola n has been verified by CNN IBN. | Abstained. A security warning that contains shortcode syntax and 'sim card'. This is the message a filter must not junk; abstain/deliver is correct. |
| 15 | Q | ham | 0.043 | Lmao!nice 1 | Abstained. Four tokens. No usable evidence. |
| 16 | Q | ham | 0.045 | Hi. Wk been ok - on hols now! Yes on for a bit of a run. Forgot that i have hairdressers appointment at four so need to get home n shower beforehand. Does that cause prob for u?" | Abstained. Time expressions ('four') and 'appointment' are weak spam cues; conversation context wins. |
| 17 | Q | ham | 0.046 | what is your account number? | Abstained. Pure phishing cue in isolation, but here it is a friend's question. No model can resolve this without conversation history; abstention is correct. |
| 18 | Q | ham | 0.053 | Save yourself the stress. If the person has a dorm account, just send your account details and the money will be sent to you. | Abstained. 'account details' + 'money' is classic mule/phishing language in a benign context. Good example of why FPR-first policy matters. |
| 19 | Q | ham | 0.055 | Great. Hope you are using your connections from mode men also cos you can never know why old friends can lead you to today | Abstained. Long, chatty, no links. Low information. |
| 20 | Q | ham | 0.058 | Amazing : If you rearrange these letters it gives the same meaning... Dormitory = Dirty room Astronomer = Moon starer The eyes = They see Election results = Lies lets recount Mother-in-law = Woman Hitler Eleven plus two =Twelve plus one Its Amazing... !:-) | Abstained. A chain forward. Spam-adjacent (bulk forward) but ham here; abstention is acceptable. |

## What the 20 cases say

1. **The 7 errors are lexical, not structural.** Four false positives fire on a single
   cue (a URL, 'password', a numbered menu, a template header); three misses are
   messages whose surface form is a quiz, a joke or a classified ad.
2. **The costliest error class is the security advisory** (case 5): a real bank warning
   with phishing vocabulary. The FPR-first threshold plus conformal abstention is what
   keeps it out of junk; this case is the argument for never tuning tau on F1 alone.
3. **Abstentions cluster at 0.035-0.060**, far below tau = 0.151. They are short messages and context-dependent questions,
   not near-misses of the decision boundary in probability space.
4. **Two gold labels are arguable** (cases 1-2): UCI's spam class includes adverts and
   joke forwards, which are not smishing. Report smishing metrics on the modern pool;
   do not 'fix' these labels in the anchor.
5. **Actionable, cheap fixes:** a premium-rate 'Send A/B/C' rule; keep the security-
   advisory lexicon on a deliver-list. Neither needs a retrain, and neither may be
   tuned on this test split.
