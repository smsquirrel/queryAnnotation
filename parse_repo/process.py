"""
Usage:
    process.py [options] INPUT_DIR OUTPUT_DIR

Options:
    -h --help
    --language LANGUAGE             Language
    --processes PROCESSES           # of processes to use [default: 16]
    --license-filter FILE           License metadata to filter, every row contains [nwo, license, language, score] (e.g. ['pandas-dev/pandas', 'bsd-3-clause', 'Python', 0.9997])
    --tree-sitter-build FILE        [default: /src/build/py-tree-sitter-languages.so]
"""
import functools
from multiprocessing import Pool
import pickle
import os  
import sys 
import builtins 
from os import PathLike
from typing import Optional, Tuple, Type, List, Dict, Any

from docopt import docopt
import pandas as pd
from tree_sitter import Language, Parser

from language_data import LANGUAGE_METADATA
from parsers.language_parser import LanguageParser, tokenize_docstring
from utils import flatten, walk

class DataProcessor:

    PARSER = Parser()

    def __init__(self, language: str, language_parser: Type[LanguageParser]):
        self.language = language
        self.language_parser = language_parser
        self.builtin_functions = self.get_builtin_function()

    def get_builtin_function(self):
        # Retrieve the list of all functions in the os module
        os_functions = [function for function in dir(os) if callable(getattr(os, function))]

        # Retrieve the list of all functions in the sys module
        sys_functions = [function for function in dir(sys) if callable(getattr(sys, function))]

        # Retrieve the list of all built-in functions
        builtin_functions = [function for function in dir(builtins) if callable(getattr(builtins, function))]

        # Combine all functions into a single list
        all_functions = os_functions + sys_functions + builtin_functions

        # Deduplicate the list in case of any overlaps
        unique_functions = list(set(all_functions))

        # Sort the list of functions alphabetically
        unique_functions.sort()
        return unique_functions

    def get_builtin_type_function(self):
        bultin_type_calls = dir(list)+dir(dict)+dir(str)+dir(set)+dir(tuple)+dir(range)+dir(bytes)+dir(bytearray)+dir(memoryview)+dir(frozenset)
        bultin_type_calls = [call for call in bultin_type_calls if not call.startswith("__")]
        # print(len(bultin_type_calls), bultin_type_calls[0])
        return bultin_type_calls

    def process_dee(self, nwo, ext, tmp_dir, sha) -> List[Dict[str, Any]]:
        # Process dependees (libraries) to get function implementations
        indexes = []
        files = walk(tmp_dir, ext)

        for f in files:
            definitions = self.get_function_definitions(f)
            if definitions is None:
                continue

            nwo, path, functions = definitions
            indexes.extend((self.extract_function_data(func, nwo, path, sha) for func in functions if (len(func['function_tokens']) > 1 and not (func["identifier"].startswith("__") and func["identifier"].endswith("__")))))
        return indexes

    def process_dent(self, nwo, ext, library_candidates, tmp_dir, sha, repo_dirs) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
        dents = []
        edges = []
        third_party_calls = []
        bultin_type_calls = self.get_builtin_type_function()
        if nwo is None:
            return dents, edges

        files = walk(tmp_dir, ext)
        # sha = None

        for f in files:
            context_and_calls = self.get_context_and_function_calls(f)  
            if context_and_calls is None:
                continue
            
            nwo, path, contexts, call_dicts = context_and_calls

            for main_function, fucntion_calls_and_vars in call_dicts.items():
                calls, function_vars = fucntion_calls_and_vars["calls"], fucntion_calls_and_vars["vars"]

                for call in calls:
                    if ("." in call['identifier'] and call['identifier'].split(".")[0] not in contexts and call['identifier'].split(".")[-1] in self.builtin_functions) or call['identifier'] in self.builtin_functions:
                        continue 
                    repo_flag = True
                    for depended_library_function in library_candidates:
                        if (len(call['identifier'].split('.')) > 1 and call['identifier'].split('.')[0] in function_vars and call['identifier'].split('.')[-1] in bultin_type_calls):
                            break

                        if (call['identifier'].split(".")[-1] == depended_library_function['identifier'].split(".")[-1]):
                            repo_flag = False
                            dent = {
                                'nwo': nwo,
                                'sha': sha,
                                'path': path,
                                'language': self.language,
                                "main_function": main_function,
                                'identifier': call['identifier'],
                                'argument_list': call['argument_list'],
                                'url': 'https://github.com/{}/blob/{}/{}#L{}-L{}'.format(nwo, sha, path,
                                                                                        call['start_point'][0] + 1,
                                                                                        call['end_point'][0] + 1)
                            }
                            dents.append(dent)
                            edges.append((dent['url'], depended_library_function['url']))
                            break   
                    
                    contexts = {import_context: from_context for import_context, from_context in contexts.items() if from_context is None or len(from_context.split('.')) == 1 or from_context.split('.')[0] not in repo_dirs}
                    if repo_flag and len(call["identifier"]) >= 3 and call['identifier'].split('.')[0] != "self" and call['identifier'].split('.')[0] in contexts and (contexts[call['identifier'].split('.')[0]] == None or (contexts[call['identifier'].split('.')[0]][0] != '.')):  
                        if contexts[call['identifier'].split('.')[0]] is not None:
                            call['identifier'] = call['identifier'].replace(call['identifier'].split('.')[0], contexts[call['identifier'].split('.')[0]])
                        dent = {
                            'nwo': nwo,
                            'sha': sha,
                            'path': path,
                            'language': self.language,
                            "main_function": main_function,
                            'identifier': call['identifier'],
                            'argument_list': call['argument_list'],
                        }
                        third_party_calls.append(dent)
        return dents, edges, third_party_calls

    def process_single_file(self, filepath: PathLike) -> List[Dict[str, Any]]:
        definitions = self.get_function_definitions(filepath)
        if definitions is None:
            return []
        _, _, functions = definitions

        return [self.extract_function_data(func, '', '', '') for func in functions if len(func['function_tokens']) > 1]

    def extract_function_data(self, function: Dict[str, Any], nwo, path: str, sha: str):
        """Extract function data from a function dictionary."""
        return {
            'nwo': nwo,
            'sha': sha,
            'path': path,
            'language': self.language,
            'identifier': function['identifier'],
            'parameters': function.get('parameters', ''),
            'argument_list': function.get('argument_list', ''),
            'return_statement': function.get('return_statement', ''),
            'docstring': function['docstring'].strip(),
            'docstring_summary': function['docstring_summary'].strip(),
            'docstring_tokens': tokenize_docstring(function['docstring_summary']),
            'function': function['function'].strip(),
            'function_tokens': function['function_tokens'],
            'url': 'https://github.com/{}/blob/{}/{}#L{}-L{}'.format(nwo, sha, path, function['start_point'][0] + 1,
                                                                     function['end_point'][0] + 1)
        }

    def get_function_name(self, tree):
        for child in tree.children:
            if child.type == "identifier":
                return child.text.decode("utf8")

    def get_context_and_function_calls(self, filepath: str) -> Optional[Tuple[str, str, List, List]]:
        nwo = '/'.join(filepath.split('/')[1:3])
        # path = '/'.join(filepath.split('/')[5:])
        path = '/'.join(filepath.split('/')[3:])
        if any(fp in path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None
        try:
            with open(filepath) as source_code:
                blob = source_code.read()
            tree = DataProcessor.PARSER.parse(blob.encode())
            # function_nodes = self.language_parser.get_definition_node(tree, blob)
            calls = {}
            functions, function_nodes = self.language_parser.get_definition(tree, blob, return_node=True)
            for function, function_node in zip(functions, function_nodes):
                function_context = function_node.text.decode("utf-8")
                function_calls_and_vars = {}
                function_calls_and_vars["calls"] = self.language_parser.get_calls_from_function(DataProcessor.PARSER.parse(function_context.encode()), function_context)
                function_calls_and_vars["vars"] = self.language_parser.extract_parameters_and_variables(DataProcessor.PARSER.parse(function_context.encode()))
                calls[function["identifier"]] = function_calls_and_vars
            return (nwo, path, self.language_parser.extract_imports(tree), calls)
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError):
            return None

    def get_function_definitions(self, filepath: str) -> Optional[Tuple[str, str, List]]:
        nwo = '/'.join(filepath.split('/')[5:7])
        path = '/'.join(filepath.split('/')[3:])
        print(f"start get function define : {filepath}")
        if any(fp in path.lower() for fp in self.language_parser.FILTER_PATHS):
            return None
        try:
            with open(filepath) as source_code:
                blob = source_code.read()
            tree = DataProcessor.PARSER.parse(blob.encode())
            return (nwo, path, self.language_parser.get_definition(tree, blob))
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError, ValueError, OSError):
            print("get function definitions error!")
            return None


if __name__ == '__main__':
    args = docopt(__doc__)

    repository_dependencies = pd.read_csv(args['INPUT_DIR'] + 'repository_dependencies-1.4.0-2018-12-22.csv', index_col=False)
    projects = pd.read_csv(args['INPUT_DIR'] + 'projects_with_repository_fields-1.4.0-2018-12-22.csv', index_col=False)

    repository_dependencies['Manifest Platform'] = repository_dependencies['Manifest Platform'].apply(lambda x: x.lower())
    id_to_nwo = {project['ID']: project['Repository Name with Owner'] for project in projects[['ID', 'Repository Name with Owner']].dropna().to_dict(orient='records')}
    nwo_to_name = {project['Repository Name with Owner']: project['Name'] for project in projects[['Repository Name with Owner', 'Name']].dropna().to_dict(orient='records')}

    filtered = repository_dependencies[(repository_dependencies['Host Type'] == 'GitHub') & (repository_dependencies['Manifest Platform'] == LANGUAGE_METADATA[args['--language']]['platform'])][['Repository Name with Owner', 'Dependency Project ID']].dropna().to_dict(orient='records')

    dependency_pairs = [(rd['Repository Name with Owner'], id_to_nwo[int(rd['Dependency Project ID'])])
                        for rd in filtered if int(rd['Dependency Project ID']) in id_to_nwo]

    dependency_pairs = list(set(dependency_pairs))

    dents, dees = zip(*dependency_pairs)
    # dents = list(set(dents))
    dees = list(set(dees))

    DataProcessor.PARSER.set_language(Language(args['--tree-sitter-build'], args['--language']))

    processor = DataProcessor(language=args['--language'],
                              language_parser=LANGUAGE_METADATA[args['--language']]['language_parser'])

    with Pool(processes=int(args['--processes'])) as pool:
        output = pool.imap_unordered(functools.partial(processor.process_dee,
                                                       ext=LANGUAGE_METADATA[args['--language']]['ext']),
                                     dees)

    definitions = list(flatten(output))
    with open(args['OUTPUT_DIR'] + '{}_definitions.pkl'.format(args['--language']), 'wb') as f:
        pickle.dump(definitions, f)

    license_filter_file = args.get('--license-filter')
    if license_filter_file is not None:
        with open(license_filter_file, 'rb') as f:
            license_filter = pickle.load(f)
        valid_nwos = dict([(l[0], l[3]) for l in license_filter])

        # Sort function definitions with repository popularity
        definitions = [dict(list(d.items()) + [('score', valid_nwos[d['nwo']])]) for d in definitions if d['nwo'] in valid_nwos]
        definitions = sorted(definitions, key=lambda x: -x['score'])

        # dedupe
        seen = set()
        filtered = []
        for d in definitions:
            if ' '.join(d['function_tokens']) not in seen:
                filtered.append(d)
                seen.add(' '.join(d['function_tokens']))

        dd = DuplicateDetector(min_num_tokens_per_document=10)
        filter_mask = [dd.add_file(id=idx,
                                   tokens=d['function_tokens'],
                                   language=d['language']) for idx, d in enumerate(filtered)]
        exclusion_set = dd.compute_ids_to_exclude()
        exclusion_mask = [idx not in exclusion_set for idx, _ in enumerate(filtered)]
        filtered = [d for idx, d in enumerate(filtered) if filter_mask[idx] & exclusion_mask[idx]]

        with open(args['OUTPUT_DIR'] + '{}_dedupe_definitions.pkl'.format(args['--language']), 'wb') as f:
            pickle.dump(filtered, f)


