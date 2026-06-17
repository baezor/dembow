# Where to find more reggaeton MIDI

Dembow's quality is limited mostly by data — ~76 short files is small for a
Transformer. More clean reggaeton/dembow MIDI is the single biggest improvement
you can make. Drop new `.mid` / `.midi` files into this folder and retrain.

## What helps most
- **Reggaeton / dembow / Latin-trap MIDI** with a real drum track on channel 10
  (the dembow kick/snare is the genre's signature).
- **Multi-track arrangements** (drums + bass + melody) — Dembow groups parts into
  drums / bass / mid / high, so layered files teach it more than single melodies.
- **4/4, straightforward tempo.** Quantized files quantize cleanly to the 16th grid.

## Free / open MIDI sources
- **FreeMIDI.org**, **BitMidi.com**, **MidiWorld** — large general libraries; search
  artist names (Daddy Yankee, Don Omar, Wisin & Yandel, Tego Calderón, Aventura).
- **The Lakh MIDI Dataset (LMD)** — ~176k MIDI files; filter to Latin/reggaeton by
  matching titles/artists. Great for bulk augmentation.
- **MetaMIDI Dataset** — large, with genre metadata you can filter on.
- **Groove MIDI Dataset (Magenta)** — expressive *drum* performances; excellent for
  teaching the groove even though it isn't reggaeton-specific.
- **Hooktheory / TheoryTab** — chord+melody data (export to MIDI) for harmony.

## Make your own
- **Transcribe audio to MIDI** with [Spotify Basic Pitch](https://basicpitch.spotify.com/)
  or [Magenta MT3](https://github.com/magenta/mt3) from reggaeton stems/acapellas.
- **Export from a DAW** (Ableton, FL Studio, Logic) — reggaeton MIDI packs are widely
  sold; bounce the MIDI clips out.
- **Program dembow patterns** by hand in any DAW and export.

## Cleaning tips
- Keep drums on channel 10 (MIDI channel index 9) so the tokenizer routes them as drums.
- Remove long silent intros/outros; Dembow trains on bars of actual music.
- One song per file; quantize to 16ths if the timing is loose.

## ⚠️ Licensing
MIDI transcriptions of copyrighted songs carry the rights of the underlying
composition. Use them for **personal experimentation / research**. Don't
redistribute copyrighted MIDI or publish generated tracks commercially without
clearing rights. Prefer openly licensed datasets (LMD, Groove MIDI, MetaMIDI) for
anything you intend to share.
