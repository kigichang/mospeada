pub fn conv_pth_to_safetensors<P: AsRef<std::path::Path>>(
    pth: P,
    dest: P,
) -> candle_core::Result<()> {
    let pth_vec = candle_core::pickle::read_all(pth)?;

    let mut tensor_map = std::collections::HashMap::new();

    for item in pth_vec {
        tensor_map.insert(item.0, item.1);
    }

    candle_core::safetensors::save(&tensor_map, dest)
}

pub(crate) fn print_vec1<T: std::fmt::Debug>(v: &[T]) {
    if v.len() <= 6 {
        println!("{:?}", v);
        return;
    }
    print!("[");
    for i in 0..3 {
        print!("{:?}, ", v[i]);
    }
    print!("...");
    for i in v.len() - 3..v.len() {
        print!(", {:?}", v[i]);
    }
    print!("]");
}

pub(crate) fn println_vec1<T: std::fmt::Debug>(v: &[T]) {
    print_vec1(v);
    println!();
}

pub(crate) fn print_vec2<T: std::fmt::Debug>(v: &[Vec<T>]) {
    match v.len() {
        0 => {
            println!("[[]]");
            return;
        }
        1 => {
            print!("[");
            print_vec1(&v[0]);
            println!("]");
            return;
        }
        _ => {}
    }

    print!("[");
    print_vec1(&v[0]);
    println!(",");
    if v.len() <= 6 {
        for i in 1..v.len() - 1 {
            print!("  ");
            print_vec1(&v[i]);
            println!(",");
        }
    } else {
        for i in 1..3 {
            print!("  ");
            print_vec1(&v[i]);
            println!(",");
        }
        println!("  ...,");
        for i in v.len() - 3..v.len() - 1 {
            print!("  ");
            print_vec1(&v[i]);
            println!(",");
        }
    }
    print!("  ");
    print_vec1(&v[v.len() - 1]);
    print!("]");
}

pub(crate) fn println_vec2<T: std::fmt::Debug>(v: &[Vec<T>]) {
    print_vec2(v);
    println!();
}

pub(crate) fn print_vec3<T: std::fmt::Debug>(v: &[Vec<Vec<T>>]) {
    match v.len() {
        0 => {
            println!("[[[]]]");
            return;
        }
        1 => {
            print!("[");
            print_vec2(&v[0]);
            println!("]");
            return;
        }
        _ => {}
    }
    print!("[");
    print_vec2(&v[0]);
    println!(",");
    if v.len() <= 6 {
        for i in 1..v.len() - 1 {
            print!(" ");
            print_vec2(&v[i]);
            println!(",");
        }
    } else {
        for i in 1..3 {
            print!(" ");
            print_vec2(&v[i]);
            println!(",");
        }
        println!(" ...,");
        for i in (v.len() - 3)..(v.len() - 1) {
            print!(" ");
            print_vec2(&v[i]);
            println!(",");
        }
    }
    print!(" ");
    print_vec2(&v[v.len() - 1]);
    print!("]");
}

pub(crate) fn println_vec3<T: std::fmt::Debug>(v: &[Vec<Vec<T>>]) {
    print_vec3(v);
    println!();
}

pub fn print_tensor<S: candle_core::WithDType + std::fmt::Debug>(
    t: &candle_core::Tensor,
) -> candle_core::Result<()> {
    match t.rank() {
        0 => println!("{:?}", t.to_scalar::<S>()?),
        1 => println_vec1(&t.to_vec1::<S>()?),
        2 => println_vec2(&t.to_vec2::<S>()?),
        3 => println_vec3(&t.to_vec3::<S>()?),
        _ => println!("{:?}", t.shape()),
    }

    Ok(())
}
