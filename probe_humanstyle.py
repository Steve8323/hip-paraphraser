"""Probe: does desklib rate a HUMAN-STYLE rewrite of an AI essay higher than the
AI essay? If not, retraining on AI->human pairs can't help on essay text and the
detector is the real bottleneck. (a = my AI essay, h = my human-style rewrite.)"""
import json
import detector_proxy

AI = [c.strip() for c in open("sources_large.txt").read().split("\n\n") if c.strip()]

HUMAN = [
    # remote work
    "Remote work has really changed how we think about jobs, and not just the where "
    "but what 'getting things done' even means anymore. Over the last few years just "
    "about every industry has gone flexible, and honestly it makes sense, since talent "
    "isn't stuck in one city now. There are obvious upsides. No soul-crushing commute, "
    "more time for your actual life, and companies can hire from basically anywhere. But "
    "it's not all rosy. Keeping a team feeling like a team is hard when nobody shares a "
    "hallway, and people drift into isolation if you're not watching for it. You need "
    "decent tools, and more than that you need to actually trust people. Where it lands, "
    "I think, is that the companies that win won't be the fully-remote ones or the "
    "back-to-the-office ones. They'll be the ones that figure out the messy middle.",
    # climate
    "Climate change is probably the biggest mess we'll have to deal with this century, "
    "and it's not the kind of thing one country fixes on its own. The science isn't "
    "really up for debate at this point. We're burning fossil fuels, the planet's "
    "heating up, and we're already seeing the fallout, from weirder weather to higher "
    "seas to ecosystems falling apart. There's no single fix. On one side, governments "
    "and industry have to actually move on clean energy and stop dragging their feet on "
    "emissions. On the other, communities need to get ready for the damage that's "
    "already baked in, building defenses and looking out for the people who get hit "
    "hardest. In the end it comes down to cooperation on a scale we've never really "
    "pulled off, plus being honest about how we treat the planet.",
    # AI education
    "AI could genuinely shake up education, and mostly for the better. The pitch is "
    "simple: software that adapts to each kid, gives feedback right away, and doesn't "
    "get tired. For students who'd never get a private tutor, that's a big deal, and it "
    "could help close gaps that have been around forever. It'd also take some of the "
    "grunt work off teachers so they can do the parts only humans can do. That said, I'm "
    "not totally sold without some caveats. Who's got the data, and what are they doing "
    "with it? What happens when the algorithm's biased? And are we quietly teaching kids "
    "to stop thinking for themselves? Schools will have to be careful here. The goal "
    "should be tools that back up teachers, not ones that replace them.",
]

print(f"{'topic':14}{'P(hum) AI':>11}{'P(hum) human-style':>20}{'delta':>9}")
rows = []
for name, a, h in zip(["remote", "climate", "education"], AI, HUMAN):
    pa, ph = detector_proxy.human_prob(a), detector_proxy.human_prob(h)
    rows.append((name, pa, ph))
    print(f"{name:14}{pa:>11.3f}{ph:>20.3f}{ph-pa:>+9.3f}")
print(f"\nmean delta P(human) = {sum(h-a for _,a,h in rows)/len(rows):+.3f}")
