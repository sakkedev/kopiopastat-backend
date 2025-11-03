translations = {
    "fi": {
        "entry_not_found": "Ei löydy :DD",
        "invalid_range": "Voi viddu virhe :DD magsimissaan 100 voi valida :DD midä vidduu sä deed :D",
        "request_timed_out": "Byyndö aigagadgaisdu :DD",
        "content_must_be_5_250000_chars": "Sisällön on oldava 5-250000 merggiä bidgä :DD älä girjoda romaania :D",
        "title_must_be_1_128_chars": "Odsigon on oldava 1-128 merggiä bidgä :DD",
        "title_already_exists": "Odsiggo löydyy jo :DD",
        "edit_successful": "MUOGADDU :D EBIN :--D",
        "new_entry_created": "Uusi kopiopasta on syntynyt!",
        "order_index_out_of_range": "Indeksiä ei löyty :("
    }
}

def translate(key, lang="fi"):
    return translations.get(lang, {}).get(key, key)
