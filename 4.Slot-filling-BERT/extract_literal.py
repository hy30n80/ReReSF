import json
import re
import argparse
from tqdm import tqdm

# Slot-Filling 학습 및 평가하기 위한 데이터를 구축하기 위한 코드로, 
# Train Set 과 Test Set 에 대해서 각각 채워야 할 LITERAL 정보들을 추출하는 코드
# Masked Cypher 와 Gold_Cypher 를 비교함으로써, LITERAL_Q, LITERAL_C, LITERAL_V 를 추출

def parse_all_cypher_patterns_final_fixed(data_item):
    """
    기존 로직을 유지하면서 "answers" 필드를 추가하도록 수정된 버전.
    """
    masked_cypher = data_item.get('masked_cypher', '')
    gold_cypher = data_item.get('gold_cypher', '')


    # 1. LITERAL_C
    literal_count = masked_cypher.count('[LITERAL]')
    data_item['LITERAL_C'] = literal_count
    if literal_count == 0:
        data_item['LITERAL_V'] = []
        data_item['LITERAL_Q'] = []
        # --- [추가된 부분] answers 필드 초기화 ---
        data_item['answers'] = [] 
        return data_item

    # 2. LITERAL_V
    parts = re.split(r"('\[LITERAL\]'|\[LITERAL\])", masked_cypher)
    pattern_parts = []
    for part in parts:
        if part == "'[LITERAL]'":
            pattern_parts.append("'(.*?)'")
        elif part == "[LITERAL]":
            pattern_parts.append("(.*?)")
        else:
            pattern_parts.append(re.escape(part))
    pattern_for_v = "".join(pattern_parts)
    match_v = re.search(pattern_for_v, gold_cypher, re.DOTALL)
    data_item['LITERAL_V'] = list(match_v.groups()) if match_v else []

    # 3. LITERAL_Q
    node_aliases = {alias: label for alias, label in re.findall(r'\((\w+):(\w+)\)', masked_cypher)}
    rel_aliases = {alias: label for alias, label in re.findall(r'\[(\w+):(\w+)\]', masked_cypher)}
    alias_map = {**node_aliases, **rel_aliases}

    raw_matches = []

    # 3-2. 모든 LITERAL 패턴 정의
    patterns = {
        "prop_in_node": r":(\w+)\s*{\s*(\w+)\s*:\s*'\[LITERAL\]'",
        "prop_op_unquoted": r"(\w+)\.(\w+)\s*(?:[=><]|<=|>=|<>)\s*\[LITERAL\]",
        "prop_op_quoted": r"(\w+)\.(\w+)\s*(?:=|<>)\s*'\[LITERAL\]'",
        "quoted_in_collection": r"'\[LITERAL\]'\s+IN\s+(\w+)\.(\w+)",
        "not_quoted_in_collection": r"NOT\s+'\[LITERAL\]'\s+IN\s+(\w+)\.(\w+)",
        "prop_op_date_quoted": r"(\w+)\.(\w+)\s*(?:[=><]|<=|>=|<>)\s*date\('\[LITERAL\]'\)",
    }

    # 3-3. 모든 잠재적 매칭을 찾아서 위치 정보와 함께 저장
    for p_name, p_regex in patterns.items():
        for match in re.finditer(p_regex, masked_cypher):
            if p_name == "prop_in_node":
                label, prop = match.groups()
            else:
                alias, prop = match.groups()
                label = alias_map.get(alias, alias)
            question = f"What is the {label} {prop}?"
            raw_matches.append({'start': match.start(), 'end': match.end(), 'question': question})

    # 3-4. 중복 매칭 필터링 로직
    raw_matches.sort(key=lambda m: (m['start'], -m['end']))
    
    filtered_matches = []
    last_end = -1
    for match in raw_matches:
        if match['start'] >= last_end:
            filtered_matches.append(match)
            last_end = match['end']

    # 3-5. 최종적으로 필터링된 질문 리스트 생성
    filtered_matches.sort(key=lambda m: m['start'])
    data_item['LITERAL_Q'] = [match['question'] for match in filtered_matches]

    # 4. "answers" 필드 생성
    # nl_question을 context로 사용하고, 추출된 LITERAL_V를 answer text로 사용합니다.
    context = data_item.get('nl_question', '')
    answer_texts = data_item.get('LITERAL_V', [])
    answers_with_positions = []

    if context and answer_texts: # context와 answer_texts가 모두 존재할 때만 실행
        for text in answer_texts:
            # context에서 answer 텍스트의 시작 위치(start_char)를 찾습니다.
            start_char = context.find(text)
            
            if start_char != -1:
                answers_with_positions.append({
                    "text": text,
                    "start_char": start_char
                })
            else:
                pass 

    data_item['answers'] = answers_with_positions

    return data_item


def process_jsonl_file(input_path, output_path):
    """
    JSON 파일(배열 형식) 또는 JSONL 파일을 처리하고, 결과를 새 JSONL 파일에 저장합니다.
    """
    print(f"파일 처리를 시작합니다: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except FileNotFoundError:
        print(f"에러: 입력 파일 '{input_path}'을 찾을 수 없습니다.")
        return

    processed_count = 0
    
    try:
        # JSON 배열 형식인지 확인
        if content.startswith('[') and content.endswith(']'):
            print("JSON 배열 형식으로 처리합니다.")
            data_list = json.loads(content)
            total_items = len(data_list)
            
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for i, data in enumerate(tqdm(data_list, total=total_items, desc="Processing file")):
                    try:
                        # 파싱 함수에 전달하기 전에 gold_cypher의 이스케이프 문자를 먼저 제거합니다.
                        if 'gold_cypher' in data and isinstance(data['gold_cypher'], str):
                            data['gold_cypher'] = data['gold_cypher'].replace("\\'", "'")

                        # data.copy()를 전달하여 원본 데이터에 영향을 주지 않도록 합니다.
                        processed_data = parse_all_cypher_patterns_final_fixed(data.copy())
                        outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    except Exception as e:
                        qid = data.get('qid', f'Item_{i}') if isinstance(data, dict) else f'Item_{i}'
                        print(f"\n에러 발생: Item {i+1}, qid: {qid} 처리 중 예외가 발생했습니다: {e}")
                        continue
        else:
            # JSONL 형식으로 처리
            print("JSONL 형식으로 처리합니다.")
            lines = content.split('\n')
            total_lines = len(lines)
            
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line_num, line in enumerate(tqdm(lines, total=total_lines, desc="Processing file")):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)

                        # 파싱 함수에 전달하기 전에 gold_cypher의 이스케이프 문자를 먼저 제거합니다.
                        if 'gold_cypher' in data and isinstance(data['gold_cypher'], str):
                            data['gold_cypher'] = data['gold_cypher'].replace("\\'", "'")

                        # data.copy()를 전달하여 원본 데이터에 영향을 주지 않도록 합니다.
                        processed_data = parse_all_cypher_patterns_final_fixed(data.copy())
                        outfile.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                        processed_count += 1
                    except json.JSONDecodeError:
                        print(f"경고: {line_num+1}번째 줄에서 JSON 파싱 에러가 발생했습니다. 해당 줄은 건너뜁니다.")
                    except Exception as e:
                        try:
                            qid = json.loads(line).get('qid', 'N/A') if isinstance(json.loads(line), dict) else 'N/A'
                        except:
                            qid = 'N/A'
                        print(f"\n에러 발생: Line {line_num+1}, qid: {qid} 처리 중 예외가 발생했습니다: {e}")
                        continue
    except Exception as e:
        print(f"파일 처리 중 예상치 못한 에러가 발생했습니다: {e}")
        return

    print("-" * 30)
    print(f"총 {processed_count}개의 데이터를 처리했습니다.")
    print(f"결과가 다음 파일에 저장되었습니다: {output_path}")
    print("-" * 30)

from collections import Counter

def validate_literal_counts(file_path):
    """
    주어진 JSONL 파일의 데이터 일관성을 검증하고,
    LITERAL_C 값에 따른 데이터 개수 통계를 함께 출력합니다.
    """
    print(f"검증을 시작합니다: {file_path}")
    
    invalid_entries = []
    line_number = 0
    # LITERAL_C 값의 개수를 세기 위한 Counter 객체
    literal_c_counter = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_number += 1
                try:
                    data = json.loads(line)
                    
                    # --- 1. 개수 일치 여부 검증 (기존 로직) ---
                    literal_c = data.get('LITERAL_C', 0)
                    len_v = len(data.get('LITERAL_V', []))
                    len_q = len(data.get('LITERAL_Q', []))
                    
                    if not (literal_c == len_v and literal_c == len_q):
                        invalid_entries.append({
                            "line": line_number,
                            "qid": data.get("qid", "N/A"),
                            "LITERAL_C": literal_c,
                            "len(LITERAL_V)": len_v,
                            "len(LITERAL_Q)": len_q
                        })
                    
                    # --- 2. LITERAL_C 값별 개수 집계 (기능 추가) ---
                    literal_c_counter[literal_c] += 1
                        
                except json.JSONDecodeError:
                    print(f"경고: {line_number}번째 줄에서 JSON 파싱 에러가 발생했습니다.")

    except FileNotFoundError:
        print(f"에러: 검증할 파일 '{file_path}'을 찾을 수 없습니다.")
        return

    # --- 결과 출력 ---

    # 1. LITERAL_C 값별 개수 통계 출력
    print("-" * 30)
    print("--- LITERAL_C 별 데이터 개수 통계 ---")
    if not literal_c_counter:
        print("  처리된 데이터가 없습니다.")
    else:
        # LITERAL_C 값을 기준으로 정렬하여 출력
        for c_value, count in sorted(literal_c_counter.items()):
            print(f"  - LITERAL_C = {c_value}: {count}개")
    
    # 2. 불일치 검증 결과 출력
    print("-" * 30)
    if not invalid_entries:
        print(f"✅ 검증 완료: 총 {line_number}개의 모든 데이터가 일관성을 가집니다.")
    else:
        print(f"❌ 검증 실패: 총 {line_number}개의 데이터 중 {len(invalid_entries)}개에서 불일치가 발견되었습니다.")
        for entry in invalid_entries:
            print(
                f"  - [Line: {entry['line']}, qid: {entry['qid']}] -> "
                f"C: {entry['LITERAL_C']}, "
                f"V_len: {entry['len(LITERAL_V)']}, "
                f"Q_len: {entry['len(LITERAL_Q)']}"
            )
    print("-" * 30)



def run_cypher_reconstruction_verification(file_path):
    """
    주어진 JSONL 파일의 각 항목에 대해 masked_cypher가 LITERAL_V로 채워졌을 때
    gold_cypher와 일치하는지 검증하고 결과를 출력합니다.

    Args:
        file_path (str): 검증할 JSONL 파일의 경로.
    """
    total_entries = 0
    matches = 0
    mismatches = 0
    mismatch_details = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                total_entries += 1

                qid = data.get('qid', f'Unknown_QID_Line_{line_num}')
                masked_cypher = data.get('masked_cypher')
                literal_v = data.get('LITERAL_V', [])
                gold_cypher = data.get('gold_cypher')

                reconstructed_cypher = masked_cypher
                for literal in literal_v:
                    reconstructed_cypher = reconstructed_cypher.replace('[LITERAL]', literal, 1)

                if reconstructed_cypher == gold_cypher:
                    # print(reconstructed_cypher)
                    # print(gold_cypher)
                    matches += 1
                else:
                    mismatches += 1
                    mismatch_details.append({
                        'qid': qid,
                        'expected_gold_cypher': gold_cypher,
                        'reconstructed_cypher': reconstructed_cypher,
                        'literal_v': literal_v
                    })
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
        return
    except json.JSONDecodeError as e:
        print(f"오류: '{file_path}' 파일의 JSON 형식이 잘못되었습니다. 문제가 발생한 줄: {e}")
        return
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
        return

    print(f"\n--- '{file_path}' 파일 검증 결과 ---")
    print(f"처리된 총 항목 수: {total_entries}")
    print(f"성공적인 재구성 (일치): {matches}")
    print(f"실패한 재구성 (불일치): {mismatches}")

    if mismatches > 0:
        print("\n--- 불일치 세부 정보 ---")
        for detail in mismatch_details:
            print(f"  QID: {detail.get('qid')}")
            print(f"    예상 Gold Cypher: {detail.get('expected_gold_cypher', 'N/A')}")
            print(f"    재구성된 Cypher: {detail.get('reconstructed_cypher', 'N/A')}")
            print(f"    사용된 LITERAL_V: {detail.get('literal_v', 'N/A')}")
            print("-" * 20)
    else:
        print("\n모든 항목이 gold_cypher를 성공적으로 재구성했습니다!")


# --- 스크립트 실행 ---
if __name__ == "__main__":
    
    # Argument parser 설정
    parser = argparse.ArgumentParser(description="Slot-Filling 데이터 구축을 위한 LITERAL 정보 추출")
    parser.add_argument("--input_file", type=str, required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--output_file", type=str, required=True, help="출력 JSONL 파일 경로")
    parser.add_argument("--skip_validation", action="store_true", help="검증 단계를 건너뛰기")
    
    args = parser.parse_args()
    
    """ Reranked 데이터 셋에 대해서, LITERAL_C, LITERAL,Q, LITERAL_V 추출 """
    process_jsonl_file(args.input_file, args.output_file)

    if not args.skip_validation:
        """ LITERAL_C = len(LITERAL_V) 인지 확인 """
        validate_literal_counts(args.output_file)

        """ LITERAL_V + Maksed_cypher = Gold_cypher 되는지 확인 """
        run_cypher_reconstruction_verification(args.output_file)